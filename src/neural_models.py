"""In this python module there are the models needed for the projects."""

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

if __name__ == '__main__':
    from utils import (
        complex_gaussian_matrix,
        complex_compressed_tensor,
        decompress_complex_tensor,
        sigma_given_snr,
        awgn,
    )
    from simnet import SimNet, RisLayer
else:
    from src.utils import (
        complex_gaussian_matrix,
        complex_compressed_tensor,
        decompress_complex_tensor,
        sigma_given_snr,
        awgn,
    )
    from src.simnet import SimNet, RisLayer


# ==================================================================
#
#                         MODELS DEFINITION
#
# ==================================================================


class ComplexAct(nn.Module):
    def __init__(self, act, use_phase: bool = False):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super().__init__()

        self.act = act
        self.use_phase = use_phase

    def forward(self, z):
        if self.use_phase:
            return self.act(torch.abs(z)) * torch.exp(1.0j * torch.angle(z))
        else:
            return self.act(z.real) + 1.0j * self.act(z.imag)


class MLP(nn.Module):
    """An implementation of a MLP in pytorch.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension. Default 1.
        hidden_dim : int
            The hidden layer dimension. Default 10.
        hidden_size : int
            The number of hidden layers. Default 10.

    Attributes:
        self.<name-of-argument>:
            ex. self.input_dim is where the 'input_dim' argument is stored.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_size: int,
    ):
        super().__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.hidden_dim: int = hidden_dim
        self.hidden_size: int = hidden_size

        # ================================================================
        #                         Input Layer
        # ================================================================
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )

        # ================================================================
        #                         Hidden Layers
        # ================================================================
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                )
                for _ in range(self.hidden_size)
            ]
        )

        # ================================================================
        #                         Output Layer
        # ================================================================
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Relative Encoder.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class ComplexMLP(nn.Module):
    """An implementation of a MLP in pytorch.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension. Default 1.
        hidden_dim : int
            The hidden layer dimension. Default 10.
        hidden_size : int
            The number of hidden layers. Default 10.

    Attributes:
        self.<name-of-argument>:
            ex. self.input_dim is where the 'input_dim' argument is stored.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_size: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size

        # ================================================================
        #                         Input Layer
        # ================================================================
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            ComplexAct(act=nn.GELU(), use_phase=True),
        )

        # ================================================================
        #                         Hidden Layers
        # ================================================================
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    ComplexAct(act=nn.GELU(), use_phase=True),
                )
                for _ in range(self.hidden_size)
            ]
        )

        # ================================================================
        #                         Output Layer
        # ================================================================
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Relative Encoder.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class Classifier(pl.LightningModule):
    """An implementation of a classifier using a MLP architecture in pytorch.

    Args:
        input_dim : int
            The input dimension.
        num_classes : int
            The number of classes. Default 20.
        hidden_dim : int
            The hidden layer dimension. Default 10.
        lr : float
            The learning rate. Default 1e-2.
        momentum : float
            How much momentum to apply. Default 0.9.
        nesterov : bool
            If set to True use nesterov type of momentum. Default True.
        max_lr : float
            Maximum learning rate for the scheduler. Default 1..

    Attributes:
        self.hparams["<name-of-argument>"]:
            ex. self.hparams["input_dim"] is where the 'input_dim' argument is stored.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 20,
        hidden_dim: int = 10,
        lr: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        max_lr: float = 1.0,
    ):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        self.accuracy = MulticlassAccuracy(
            num_classes=self.hparams['num_classes']
        )

        # Example input
        self.example_input_array = torch.randn(self.hparams['input_dim'])

        self.model = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.hparams['input_dim']),
            nn.Linear(self.hparams['input_dim'], self.hparams['hidden_dim']),
            nn.Tanh(),
            nn.LayerNorm(normalized_shape=self.hparams['hidden_dim']),
            nn.Linear(self.hparams['hidden_dim'], self.hparams['num_classes']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Classifier.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        x = nn.functional.normalize(x, p=2, dim=-1)
        return self.model(x)

    def configure_optimizers(self) -> dict[str, object]:
        """Define the optimizer: Stochastic Gradient Descent.

        Returns:
            dict[str, object]
                The optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return {
            'optimizer': optimizer,
        }

    def loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A convenient method to get the loss on a batch.

        Args:
            x : torch.Tensor
                The input tensor.
            y : torch.Tensor
                The original output tensor.

        Returns:
            (logits, loss) : tuple[torch.Tensor, torch.Tensor]
                The output of the MLP and the loss.
        """
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        return logits, loss

    def _shared_eval(
        self, batch: list[torch.Tensor], batch_idx: int, prefix: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A common step performend in the test and validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (logits, loss) : tuple[torch.Tensor, torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        x, y = batch
        logits, loss = self.loss(x, y)

        # Getting the predictions
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)
        self.log(f'{prefix}/acc_epoch', acc, on_step=False, on_epoch=True)

        return preds, loss

    def training_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """The training step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        x, y = batch
        logits, loss = self.loss(x, y)

        # Getting the predictions
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log('train/loss', loss, on_epoch=True)
        self.log('train/acc', acc, on_epoch=True)

        return loss

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        """The test step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(batch, batch_idx, 'test')
        return None

    def validation_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """The validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            preds : torch.Tensor
                The output of the network.
        """
        preds, _ = self._shared_eval(batch, batch_idx, 'valid')
        return preds

    def predict_step(
        self, batch: list[torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> torch.Tensor:
        """The predict step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.

        Returns:
            preds : torch.Tensor
                The output of the network.
        """
        x = batch[0]

        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        return preds

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class probabilities for each observation.

        Args:
            x : torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor
                Class probabilities of shape (batch_size, num_classes)
        """
        logits = self(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def entropy(
        self, x: torch.Tensor, normalized: bool = True
    ) -> torch.Tensor:
        """Compute predictive entropy per observation.

        Args:
            x : torch.Tensor
                Input tensor.
            normalized : bool
                If True, normalize by log(num_classes) to range [0, 1].

        Returns:
            torch.Tensor
                Entropy per observation (shape: [batch_size])
        """
        probs = self.predict_proba(x)
        eps = 1e-12
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)

        if normalized:
            entropy /= torch.log(
                torch.tensor(
                    self.hparams['num_classes'],
                    device=entropy.device,
                    dtype=entropy.dtype,
                )
            )
        return entropy

    def precision_weight(
        self, x: torch.Tensor, normalized: bool = True
    ) -> torch.Tensor:
        """Compute a precision-like confidence weight based on entropy.

        Args:
            x : torch.Tensor
                Input tensor.
            normalized : bool
                Whether to normalize entropy before inversion.

        Returns:
            torch.Tensor
                Precision weights (inverse of entropy), shape: [batch_size]
        """
        ent = self.entropy(x, normalized=normalized)
        weights = 1.0 / (ent + 1e-6)
        weights = weights / weights.sum()  # normalize so weights sum to 1
        return weights


class NeuralModel(pl.LightningModule):
    """An implementation of a relative encoder using a MLP architecture in pytorch.

    Args:
        input_dim : int
            The input dimension.
        output_dim : int
            The output dimension.
        antennas_transmitter : int
            The number of antennas the transmitter has.
        antennas_receiver : int
            The number of antennas the receiver has.
        hidden_dim : int
            The hidden layer dimension.
        hidden_size : int
            The number of hidden layers.
        sim_layers: int
            The number of sim layers. Default 0.
        sim_dim: int
            The number of sim elements. Default 8.
        lam : float
            The lambda of the electromagnetic wave. Default 0.125.
        snr : float
            The snr in dB of the communication channel. Set to None if unaware. Default 20.
        lr : float
            The learning rate. Default 1e-3.

    Attributes:
        self.hparams["<name-of-argument>"]:
            ex. self.hparams["input_dim"] is where the 'input_dim' argument is stored.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        antennas_transmitter: int,
        antennas_receiver: int,
        hidden_dim: int,
        hidden_size: int,
        sim_layers: int = 0,
        sim_dim: int = 8,
        lam: float = 0.125,
        layer_dist_multiplier: int = 5,
        snr: float = 20.0,
        sample: bool = False,
        lr: float = 1e-3,
    ):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        assert input_dim % 2 == 0, 'The input dimension must be even.'
        assert output_dim % 2 == 0, 'The output dimension must be even.'
        assert sim_layers >= 0, 'The number of sim layers must be positive.'
        assert isinstance(sample, bool) or sample is None, (
            'sample must be boolean or None.'
        )

        # Example input
        self.example_input_array = torch.randn(1, self.hparams['input_dim'])

        # Halve the input and output dimension
        input_dim = (input_dim + 1) // 2
        output_dim = (output_dim + 1) // 2

        if self.hparams['sim_layers'] != 0:
            self.sim = SimNet(
                [RisLayer(16, 12)]
                + [RisLayer(64, 64) for _ in range(self.hparams['sim_layers'])]
                + [RisLayer(24, 16)],
                layer_dist=self.hparams['layer_dist_multiplier']
                * self.hparams['lam'],
                wavelength=self.hparams['lam'],
                elem_area=self.hparams['lam'] ** 2 / 4,
                elem_dist=self.hparams['lam'] / 2,
            )

        self.mapper = ComplexMLP(
            input_dim,
            output_dim,
            self.hparams['hidden_dim'],
            self.hparams['hidden_size'],
        )

        # =======================================================
        #                    Channels Definition
        # =======================================================

        # Direct Channel
        self.hparams['channel_D'] = complex_gaussian_matrix(
            0,
            1,
            (
                antennas_receiver,
                antennas_transmitter,
            ),
        ).to(self.device)

        # Channel before and after the sim
        if sim_layers > 0:
            self.hparams['channel_1'] = complex_gaussian_matrix(
                0,
                1,
                (
                    sim_dim,
                    antennas_transmitter,
                ),
            ).to(self.device)
            self.hparams['channel_2'] = complex_gaussian_matrix(
                0,
                1,
                (
                    antennas_receiver,
                    sim_dim,
                ),
            ).to(self.device)

        self.type(torch.complex64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Relative Encoder.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        # x = nn.functional.normalize(x, p=2, dim=-1)

        x = complex_compressed_tensor(x.H, device=self.device).H

        # Retrieve the packets to transmit
        packets = torch.split(x, self.hparams['antennas_transmitter'], dim=1)

        # Fix the power to respect transmitting cost 1.0
        packets = [nn.functional.normalize(p, p=2, dim=-1) for p in packets]

        # Make the signal pass through the direct channel
        # z_D = self.pass_through_channel(packets, 'channel_D')

        # SIM
        if self.hparams['sim_layers'] != 0:
            # Passage through H1
            # z_sim = self.pass_through_channel(packets, 'channel_1')
            z_sim = packets

            # Passage through SIM
            z_sim = [self.sim(z) for z in z_sim]

            # Passage through H2
            # z_sim = self.pass_through_channel(z_sim, 'channel_2')

            # z = [zD + zSim for zD, zSim in zip(z_D, z_sim)]
            z = z_sim
        # else:
        #     z = z_D

        # Add white noise
        if self.hparams['snr']:
            sigma = sigma_given_snr(
                snr=self.hparams['snr'],
                signal=(
                    torch.ones(1)
                    / math.sqrt(self.hparams['antennas_transmitter'])
                ).detach(),
            )

            z = [
                i
                + awgn(
                    sigma=sigma, size=i.real.shape, device=self.device
                ).detach()
                for i in z
            ]

        z = torch.cat(z, dim=1)
        # z = torch.cat(packets, dim=1)

        # z = self.mapper(z)

        # Decompressing into Real
        return decompress_complex_tensor(z.H, device=self.device).H[
            :, : self.hparams['output_dim']
        ]

    def pass_through_channel(
        self,
        x: list[torch.tensor],
        name: str,
    ) -> torch.Tensor:
        """A usefull function to simulate the passage through a Rayleigh Channel.

        Args:
            x : list[torch.tensor]
                A list of messages to pass through the channel.
            name : str
                The specific nominative of the channel.

        Returns:
            x : list[torch.tensor]
                The list of the transmitted messages.
        """
        match self.hparams['sample']:
            case True:
                # Being True then the channel is consider dynamic
                size = self.hparams[name].shape
                channel = complex_gaussian_matrix(
                    0,
                    1,
                    size,
                ).to(self.device)

            case False:
                # Being False then the channel is consider static
                channel = self.hparams[name]

            case None:
                # Being None then the channel will not be consider
                r, _ = self.hparams[name].shape
                channel = torch.eye(r, dtype=torch.complex64).to(self.device)

        # if self.hparams['sample']:
        #     size = self.hparams[name].shape
        #     channel = complex_gaussian_matrix(
        #         0,
        #         1,
        #         size,
        #     ).to(self.device)

        # else:
        #     channel = self.hparams[name]

        x = [
            torch.einsum(
                'ab, cb -> ac',
                i,
                channel,
            )
            for i in x
        ]
        return x

    def configure_optimizers(self) -> dict[str, object]:
        """Define the optimizer: Stochastic Gradient Descent.

        Returns:
            dict[str, object]
                The optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return {
            'optimizer': optimizer,
        }

    def loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A convenient method to get the loss on a batch.

        Args:
            x : torch.Tensor
                The input tensor.
            y : torch.Tensor
                The original output tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                The output of the MLP and the loss.
        """
        y_hat = self(x)

        loss = nn.functional.mse_loss(y_hat, y)

        # Log the losses
        self.log('primal_loss', loss, on_step=True, on_epoch=True)

        return y_hat, loss

    def _shared_eval(
        self, batch: list[torch.Tensor], batch_idx: int, prefix: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A common step performend in the test and validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (y_hat, loss) : tuple[torch.Tensor, torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        x, y = batch
        y_hat, loss = self.loss(x, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)

        return y_hat, loss

    def training_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """The training step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        x, y = batch
        _, loss = self.loss(x, y)

        self.log('train/loss', loss, on_epoch=True)

        return loss

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        """The test step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(batch, batch_idx, 'test')
        return None

    def validation_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """The validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            y_hat : torch.Tensor
                The output of the network.
        """
        y_hat, _ = self._shared_eval(batch, batch_idx, 'valid')
        return y_hat

    def predict_step(
        self, batch: list[torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> torch.Tensor:
        """The predict step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.

        Returns:
            torch.Tensor
                The output of the network.
        """
        x, y = batch
        return self(x)


def main() -> None:
    """The main script loop in which we perform some sanity tests."""

    print('Start performing sanity tests...')
    print()

    # Variables definition
    input_dim = 16
    output_dim = 2
    num_classes = 2
    hidden_dim = 10
    hidden_size = 4
    antennas_transmitter = 4
    antennas_receiver = 4
    snr = 20.0

    data = torch.randn(10, input_dim)

    print()
    print('Test for Classifier...', end='\t')
    mlp = Classifier(input_dim, num_classes, hidden_dim, hidden_size)
    mlp(data)
    print('[Passed]')

    print()
    print('Test for NeuralModel...', end='\t')
    mlp = NeuralModel(
        input_dim=input_dim,
        output_dim=output_dim,
        antennas_transmitter=antennas_transmitter,
        antennas_receiver=antennas_receiver,
        hidden_dim=hidden_dim,
        hidden_size=hidden_size,
        sim_layers=2,
        snr=snr,
    )
    mlp(data)
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
