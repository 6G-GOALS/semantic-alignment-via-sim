"""A script for fitting a sim to mimic an alignment matrix A."""

# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import math
import hydra
import torch
import polars as pl

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from torch.utils.data import TensorDataset, DataLoader

from src.sim import SIMoptimizerTorch
from src.models_tasks.classification import Classifier
from src.datamodules.alignment import DataModuleAlignment
from src.alignment_utils import (
    ppfe,
    ridge_regression,
)
from src.utils import (
    complex_compressed_tensor,
    decompress_complex_tensor,
    prewhiten,
    a_inv_times_b,
    complex_gaussian_matrix,
    awgn,
    mimo_equalization,
)


# =============================================
#
#               THE MAIN LOOP
#
# =============================================


@hydra.main(
    config_path='../.conf/hydra/sim',
    config_name='settings',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """The main loop."""

    # Define some usefull paths
    CURRENT: Path = Path('.')
    MODEL_PATH: Path = CURRENT / 'models'
    RESULTS_PATH: Path = CURRENT / 'results/linear'

    # Create directories
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    # Safe way to avoid duplicate registration
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)

    # Define some variables
    lmb: float = cfg.alignment.lmb
    weighted: str = cfg.alignment.weighted
    alignment_type: str = cfg.alignment.type
    n_proto: int = int(cfg.alignment.n_proto)
    n_clusters: int = int(cfg.alignment.n_clusters)
    uuid: str = f'{cfg.seed}_{cfg.datamodule.dataset}_{cfg.alignment.type}_{cfg.channel.snr_db}_{n_proto}_{cfg.sim.layers}_{cfg.sim.wavelength}_{cfg.alignment.lmb}_{cfg.alignment.n_clusters}_{cfg.alignment.weighted}_{cfg.sim.thickness}_{cfg.sim.meta_atoms_intermediate_x}_{cfg.sim.meta_atoms_intermediate_y}_{cfg.sim.meta_atom_spacing_input_x}_{cfg.sim.meta_atom_spacing_input_y}_{cfg.sim.meta_atom_spacing_output_x}_{cfg.sim.meta_atom_spacing_output_y}_{cfg.sim.meta_atom_spacing_intermediate_x}_{cfg.sim.meta_atom_spacing_intermediate_y}_{cfg.datamodule.train_label_size}_{cfg.datamodule.grouping}_{cfg.datamodule.method}_{cfg.simulation}'

    # Define some variables
    trainer: Trainer = Trainer(
        inference_mode=True,
        enable_progress_bar=False,
        logger=False,
        accelerator=cfg.device,
    )

    # ===========================================================
    #                 Datamodule Initialization
    # ===========================================================
    datamodule: DataModuleAlignment = DataModuleAlignment(
        dataset=cfg.datamodule.dataset,
        tx_enc=cfg.models.transmitter,
        rx_enc=cfg.models.receiver,
        train_label_size=cfg.datamodule.train_label_size,
        method=cfg.datamodule.method,
        grouping=cfg.datamodule.grouping,
        batch_size=cfg.datamodule.batch_size,
        seed=cfg.seed,
    )

    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup()

    # ===========================================================
    #                  Channel Stuff
    # ===========================================================
    # Setting the seed
    seed_everything(cfg.seed, workers=True)

    n, m = datamodule.test_data.z_rx.shape
    m = int(m // 2)

    # Initializating a wireless mimo channel
    channel = complex_gaussian_matrix(0, 1, size=(m, m)).to(cfg.device)

    # Equalizing the channel
    equalizer = mimo_equalization(channel, snr_db=cfg.channel.snr_db)

    # ============================================================
    #                  Classifier Initialization
    # ============================================================
    # Define the path toweds the classifier
    clf_path: Path = (
        MODEL_PATH
        / f'classifiers/{cfg.datamodule.dataset}/{cfg.models.receiver}/seed_{cfg.seed}.ckpt'
    )

    # Load the classifier model
    clf = Classifier.load_from_checkpoint(clf_path)
    clf.to(cfg.device)
    clf.eval()

    n_classes = clf.hparams['num_classes']

    # ============================================================
    #                     Alignment Matrix
    # ============================================================

    # Complex compression
    input = complex_compressed_tensor(
        datamodule.train_data.z_tx.H, device=cfg.device
    )
    output = complex_compressed_tensor(
        datamodule.train_data.z_rx.H, device=cfg.device
    )

    # Prewhitening
    input, L_input, mean_input = prewhiten(input, device=cfg.device)
    output, L_output, mean_output = prewhiten(output, device=cfg.device)

    if cfg.alignment.weighted is not None:
        assert cfg.alignment.type == 'Linear', (
            'A weighted alignment assumes a linear alignment.'
        )

    match weighted:
        case 'Precision':
            weights = clf.precision_weight(
                datamodule.train_data.z_rx.to(cfg.device)
            ).detach()

        case 'Entropy':
            weights = clf.entropy(
                datamodule.train_data.z_rx.to(cfg.device)
            ).detach()
            weights *= 100
        case None:
            weights = None
        case _:
            raise Exception(
                'The passed weight method is not currently supported.'
            )

    match alignment_type:
        case 'Linear':
            A = ridge_regression(input, output, weights=weights, lmb=lmb)
            n_proto = None
            n_clusters = None

            if weighted is not None:
                alignment_type += ' ' + weighted

        case 'Procrustes':
            A = ridge_regression(input, output, weights=weights, lmb=lmb)

            # Get the decomposition
            U, _, Vh = torch.linalg.svd(A, full_matrices=False)

            # Procrustes solution
            A = U @ Vh

            n_proto = None
            n_clusters = None

            if weighted is not None:
                alignment_type += ' ' + weighted

        case 'PPFE':
            A = ppfe(
                input,
                output,
                output_real=datamodule.train_data.z_rx,
                n_clusters=n_clusters,
                n_proto=n_proto,
                seed=cfg.seed,
            )
            lmb = None
            weighted = None
        case _:
            raise Exception(
                'The passed alignment type is currently not supported.'
            )

    # ============================================================
    #                         SIM
    # ============================================================
    sim = SIMoptimizerTorch(
        num_intermediate_layers=cfg.sim.layers,
        num_meta_atoms_input_x=cfg.sim.meta_atoms_input_x,
        num_meta_atoms_input_y=cfg.sim.meta_atoms_input_y,
        num_meta_atoms_output_x=cfg.sim.meta_atoms_output_x,
        num_meta_atoms_output_y=cfg.sim.meta_atoms_output_y,
        num_meta_atoms_intermediate_x=cfg.sim.meta_atoms_intermediate_x,
        num_meta_atoms_intermediate_y=cfg.sim.meta_atoms_intermediate_y,
        sim_thickness=cfg.sim.thickness,
        wavelength=cfg.sim.wavelength,
        meta_atom_spacing_input_x=cfg.sim.meta_atom_spacing_input_x,
        meta_atom_spacing_input_y=cfg.sim.meta_atom_spacing_input_y,
        meta_atom_spacing_output_x=cfg.sim.meta_atom_spacing_output_x,
        meta_atom_spacing_output_y=cfg.sim.meta_atom_spacing_output_y,
        meta_atom_spacing_intermediate_x=cfg.sim.meta_atom_spacing_intermediate_x,
        meta_atom_spacing_intermediate_y=cfg.sim.meta_atom_spacing_intermediate_y,
    )

    optimized_phase_shifts, loss_history = sim.optimize_with_torch(
        A.cpu().resolve_conj().numpy(),
        max_iterations=cfg.sim.max_iters,
        lr=cfg.sim.lr,
    )

    # Analyze the optimized result
    sim._phase_shifts = optimized_phase_shifts
    optimized_G = sim._calculate_sim_propagation_G().detach().cpu().numpy()
    beta = sim.calculate_optimal_beta(
        optimized_G, A.cpu().resolve_conj().numpy()
    )

    simA = (
        torch.from_numpy(beta * optimized_G).to(torch.complex64).to(cfg.device)
    )

    # ============================================================
    #
    #                     Results
    #
    # ============================================================
    test_input = complex_compressed_tensor(
        datamodule.test_data.z_tx.H, device=cfg.device
    )
    test_input = a_inv_times_b(L_input, test_input - mean_input)

    # ============================================================
    #                       No Mismatch
    # ============================================================
    dataloader = DataLoader(
        TensorDataset(datamodule.test_data.z_rx, datamodule.test_data.labels),
        batch_size=cfg.datamodule.batch_size,
    )
    clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

    # Save Metrics
    clf_loss_nomismatch = clf_metrics['test/loss_epoch']
    clf_acc_nomismatch = clf_metrics['test/acc_epoch']

    print('Classifier Loss Original A No Mismatch:', clf_loss_nomismatch)
    print('Classifier accuracy Original A No Mismatch:', clf_acc_nomismatch)
    print()
    print()

    # ============================================================
    #            Just Alignment with Original A
    # ============================================================
    # Alignment
    y_hat = A @ test_input

    # Dewhitening
    y_hat = L_output @ y_hat + mean_output

    # Decompression
    y_hat = decompress_complex_tensor(y_hat, device=cfg.device).H

    dataloader = DataLoader(
        TensorDataset(y_hat, datamodule.test_data.labels),
        batch_size=cfg.datamodule.batch_size,
    )
    clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

    # Save Metrics
    mse_original_nomimo = torch.nn.functional.mse_loss(
        y_hat, datamodule.test_data.z_rx.to(cfg.device), reduction='mean'
    )
    clf_loss_original_nomimo = clf_metrics['test/loss_epoch']
    clf_acc_original_nomimo = clf_metrics['test/acc_epoch']

    print('Classifier Loss Original A No Mimo:', clf_loss_original_nomimo)
    print('Classifier accuracy Original A No Mimo:', clf_acc_original_nomimo)
    print('MSE Original A No Mimo:', mse_original_nomimo)
    print()
    print()

    # ============================================================
    #            Alignment with Original A + MIMO
    # ============================================================
    # Alignment
    y_hat = A @ test_input

    # Passage through channel
    y_hat = channel @ y_hat
    if cfg.channel.snr_db is not None:
        snr_linear = math.pow(10, cfg.channel.snr_db / 10)

        channel_pinv = torch.linalg.pinv(channel)

        sigma = torch.sqrt(
            torch.trace(A.H @ A)
            / (snr_linear * torch.trace(channel_pinv.H @ channel_pinv))
        ).real.item()

        print(sigma)

        # Get the AWGN
        w = awgn(sigma, size=y_hat.shape, device=cfg.device)

        # Add AWGN
        y_hat += w
    # Equalizer
    y_hat = equalizer @ y_hat

    # Dewhitening
    y_hat = L_output @ y_hat + mean_output

    # Decompression
    y_hat = decompress_complex_tensor(y_hat, device=cfg.device).H

    dataloader = DataLoader(
        TensorDataset(y_hat, datamodule.test_data.labels),
        batch_size=cfg.datamodule.batch_size,
    )
    clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

    # Save Metrics
    mse_original_mimo = torch.nn.functional.mse_loss(
        y_hat, datamodule.test_data.z_rx.to(cfg.device), reduction='mean'
    )
    clf_loss_original_mimo = clf_metrics['test/loss_epoch']
    clf_acc_original_mimo = clf_metrics['test/acc_epoch']

    print('Classifier Loss Original A Mimo:', clf_loss_original_mimo)
    print('Classifier accuracy Original A Mimo:', clf_acc_original_mimo)
    print('MSE Original A Mimo:', mse_original_mimo)
    print()
    print()

    # ============================================================
    #                  Alignment with SIM A
    # ============================================================
    # Alignment
    y_hat = simA @ test_input

    # Dewhitening
    y_hat = L_output @ y_hat + mean_output

    # Decompression
    y_hat = decompress_complex_tensor(y_hat, device=cfg.device).H

    dataloader = DataLoader(
        TensorDataset(y_hat, datamodule.test_data.labels),
        batch_size=cfg.datamodule.batch_size,
    )
    clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

    # Save Metrics
    mse_sim_nomimo = torch.nn.functional.mse_loss(
        y_hat, datamodule.test_data.z_rx.to(cfg.device), reduction='mean'
    )
    clf_loss_sim_nomimo = clf_metrics['test/loss_epoch']
    clf_acc_sim_nomimo = clf_metrics['test/acc_epoch']

    print('Classifier Loss Sim A No Mimo:', clf_loss_sim_nomimo)
    print('Classifier accuracy Sim A No Mimo:', clf_acc_sim_nomimo)
    print('MSE Sim A No Mimo:', mse_sim_nomimo)
    print()
    print()

    # ============================================================
    #            Alignment with SIM A + MIMO
    # ============================================================
    # Alignment
    y_hat = simA @ test_input

    # Passage through channel
    y_hat = channel @ y_hat
    if cfg.channel.snr_db is not None:
        # Add AWGN
        y_hat += w

    # Channel equalization
    y_hat = equalizer @ y_hat

    # Dewhitening
    y_hat = L_output @ y_hat + mean_output

    # Decompression
    y_hat = decompress_complex_tensor(y_hat, device=cfg.device).H

    dataloader = DataLoader(
        TensorDataset(y_hat, datamodule.test_data.labels),
        batch_size=cfg.datamodule.batch_size,
    )
    clf_metrics = trainer.test(model=clf, dataloaders=dataloader)[0]

    # Save Metrics
    mse_sim_mimo = torch.nn.functional.mse_loss(
        y_hat, datamodule.test_data.z_rx.to(cfg.device), reduction='mean'
    )
    clf_loss_sim_mimo = clf_metrics['test/loss_epoch']
    clf_acc_sim_mimo = clf_metrics['test/acc_epoch']

    print('Classifier Loss Sim A Mimo:', clf_loss_sim_mimo)
    print('Classifier accuracy Sim A Mimo:', clf_acc_sim_mimo)
    print('MSE Sim A Mimo:', mse_sim_mimo)
    print()
    print()

    pl.DataFrame(
        {
            'Dataset': cfg.datamodule.dataset,
            'Training Label Size': cfg.datamodule.train_label_size,
            'Grouping': cfg.datamodule.grouping,
            'Method': cfg.datamodule.method,
            'Classes': n_classes,
            'Seed': cfg.seed,
            'Alignment Type': alignment_type,
            'Number Proto': n_proto,
            'Number Clusters': n_clusters,
            'Weighted': weighted,
            'Accuracy No Mismatch': clf_acc_nomismatch,
            'Classifier Loss No Mismatch': clf_loss_nomismatch,
            'MSE Original No Mimo': mse_original_nomimo,
            'Accuracy Original No Mimo': clf_acc_original_nomimo,
            'Classifier Loss Original No Mimo': clf_loss_original_nomimo,
            'MSE Original Mimo': mse_original_mimo,
            'Accuracy Original Mimo': clf_acc_original_mimo,
            'Classifier Loss Original Mimo': clf_loss_original_mimo,
            'MSE SIM No Mimo': mse_sim_nomimo,
            'Accuracy SIM No Mimo': clf_acc_sim_nomimo,
            'Classifier Loss SIM No Mimo': clf_loss_sim_nomimo,
            'MSE SIM Mimo': mse_sim_mimo,
            'Accuracy SIM Mimo': clf_acc_sim_mimo,
            'Classifier Loss SIM Mimo': clf_loss_sim_mimo,
            'SIM Training Loss': [loss_history],
            'Receiver Model': cfg.models.transmitter,
            'Transmitter Model': cfg.models.receiver,
            'Latent Real Dim': datamodule.input_size,
            'Latent Complex Dim': (datamodule.input_size + 1) // 2,
            'Lambda': lmb,
            'SNR [dB]': cfg.channel.snr_db,
            'SIM Layers': cfg.sim.layers,
            'SIM Wavelength': cfg.sim.wavelength,
            'SIM Thickness': cfg.sim.thickness,
            'SIM Meta Atoms Intermediate X': cfg.sim.meta_atoms_intermediate_x,
            'SIM Meta Atoms Intermediate Y': cfg.sim.meta_atoms_intermediate_y,
            'SIM Meta Atoms Spacing Input X': cfg.sim.meta_atom_spacing_input_x,
            'SIM Meta Atoms Spacing Input Y': cfg.sim.meta_atom_spacing_input_y,
            'SIM Meta Atoms Spacing Output X': cfg.sim.meta_atom_spacing_output_x,
            'SIM Meta Atoms Spacing Output Y': cfg.sim.meta_atom_spacing_output_y,
            'SIM Meta Atoms Spacing Intermediate X': cfg.sim.meta_atom_spacing_intermediate_x,
            'SIM Meta Atoms Spacing Intermediate Y': cfg.sim.meta_atom_spacing_intermediate_y,
            'SIM Learning Rate': cfg.sim.lr,
            'Iterations': cfg.sim.max_iters,
            'Simulation': cfg.simulation,
        }
    ).with_columns(
        pl.col('Number Proto').cast(pl.UInt32),
        pl.col('Number Clusters').cast(pl.UInt32),
        pl.col('Lambda').cast(pl.Float32),
        pl.col('Weighted').cast(pl.String),
        pl.col('SNR [dB]').cast(pl.Float32),
    ).write_parquet(RESULTS_PATH / f'{uuid}.parquet')

    return None


if __name__ == '__main__':
    main()
