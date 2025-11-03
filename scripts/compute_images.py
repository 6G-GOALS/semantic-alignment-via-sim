"""A usefull script used to compute the needed plots.

The script expects to have the results saved in the following structure (both linear and baseline results):
|_ results/
    |_ neural/
    |   |_ r1.parquet
    |   |_ ...
    |   |_ rk.parquet
    |_ linear/
       |_ r1.parquet
       |_ ...
       |_ rk.parquet
"""

import polars as pl
from pathlib import Path
import polars.selectors as cs

import seaborn as sns
import matplotlib.pyplot as plt


def accuracy_vs_simlayers(
    df: pl.DataFrame, dataset: str, img_path: Path
) -> None:
    """ """
    filter = (
        (pl.col('Simulation') == 'accuracyVSsimlayers')
        &
        # (pl.col('Weighted').is_null()) &
        (pl.col('Dataset') == dataset)
    )

    ticks = df.filter(filter)['SIM Layers'].unique().to_list()

    original_acc = (
        df.filter(filter)
        .select('Accuracy No Mismatch')
        .unique('Accuracy No Mismatch')
        .mean()
        .item()
    )
    ax = sns.lineplot(
        df.filter(filter).drop('SIM Training Loss'),
        x='SIM Layers',
        y='Accuracy SIM Mimo',
        hue='Intermediate Layers Atoms',
        style='Alignment Type',
        markers=True,
        dashes=True,
    )
    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    layer_labels = (
        df.filter(filter)['Intermediate Layers Atoms']
        .sort(descending=True)
        .unique()
        .to_list()
    )

    alignment_labels = (
        df.filter(filter)['Alignment Type']
        .sort(descending=True)
        .unique()
        .to_list()
    )

    layer_labels = sorted(layer_labels)
    alignment_labels = sorted(alignment_labels)

    layer_handles = [handles[labels.index(cl)] for cl in layer_labels]
    alignment_handles = [handles[labels.index(cl)] for cl in alignment_labels]

    legend1 = ax.legend(
        alignment_handles,
        alignment_labels,
        title='Alignment Type',
        ncol=2,
        loc='upper center',
        # bbox_to_anchor=(0.7, 1.18),
        bbox_to_anchor=(0.37, 0.18),
        frameon=True,
        framealpha=1,
    )

    ax.legend(
        layer_handles,
        layer_labels,
        title='Intermediate Layers Atoms',
        ncol=4,
        loc='upper center',
        # bbox_to_anchor=(0.3, 1.18),
        bbox_to_anchor=(0.76, 0.18),
        frameon=True,
        framealpha=1,
    )

    ax.add_artist(legend1)

    plt.axhline(
        y=original_acc, color='gray', linestyle=':', label='No Mismatch'
    )
    plt.text(1.5, original_acc - 0.05, 'No Mismatch', color='gray')

    plt.xlabel(r'SIM Layers $L$')
    plt.ylabel('Accuracy')
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    plt.savefig(
        str(img_path / f'AccuracyVsSIMLayers_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'AccuracyVsSIMLayers_{dataset}.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    return None


def accuracy_vs_snr(df: pl.DataFrame, dataset: str, img_path: Path) -> None:
    """ """
    filter = (
        (pl.col('Simulation') == 'accuracyVSsnr')
        &
        # (pl.col('Weighted').is_null()) &
        (pl.col('Dataset') == dataset)
    )

    ticks = df.filter(filter)['SNR [dB]'].unique().to_list()

    original_acc = (
        df.filter(filter)
        .select('Accuracy No Mismatch')
        .unique('Accuracy No Mismatch')
        .mean()
        .item()
    )

    ax = sns.lineplot(
        df.filter(filter).drop('SIM Training Loss'),
        x='SNR [dB]',
        y='Accuracy SIM Mimo',
        hue='Intermediate Layers Atoms',
        style='SIM Layers',
        markers=True,
        dashes=True,
    )
    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    layer_labels = (
        df.filter(filter)['Intermediate Layers Atoms']
        # .sort(descending=True)
        .unique()
        .to_list()
    )

    num_layers_labels = (
        df.filter(filter)['SIM Layers']
        # .sort(descending=True)
        .unique()
        .to_list()
    )

    layer_labels = sorted(layer_labels)
    num_layers_labels = sorted(num_layers_labels)

    layer_handles = [handles[labels.index(cl)] for cl in layer_labels]
    num_layers_handles = [
        handles[labels.index(str(cl))] for cl in num_layers_labels
    ]

    legend1 = ax.legend(
        num_layers_handles,
        num_layers_labels,
        title=r'SIM Layers $L$',
        ncol=2,
        loc='upper center',
        # bbox_to_anchor=(0.7, 1.18),
        # bbox_to_anchor=(0.84, 0.62),
        bbox_to_anchor=(0.37, 0.18),
        frameon=True,
        framealpha=1,
    )

    ax.legend(
        layer_handles,
        layer_labels,
        title='Intermediate Layers Atoms',
        ncol=4,
        loc='upper center',
        # bbox_to_anchor=(0.3, 1.18),
        bbox_to_anchor=(0.76, 0.18),
        frameon=True,
        framealpha=1,
    )

    ax.add_artist(legend1)

    plt.axhline(
        y=original_acc, color='gray', linestyle=':', label='No Mismatch'
    )
    plt.text(-25.0, original_acc - 0.05, 'No Mismatch', color='gray')

    plt.xlabel('Signal to Noise Ratio [dB]')
    plt.ylabel('Accuracy')
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    plt.savefig(
        str(img_path / f'AccuracyVsSNR_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'AccuracyVsSNR_{dataset}.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    return None


def accuracy_original_vs_simlayers(
    df: pl.DataFrame, dataset: str, img_path: Path
) -> None:
    """ """
    filter = (
        (pl.col('Simulation') == 'accuracyVSsimlayers')
        &
        # (pl.col('Weighted').is_null()) &
        (pl.col('Dataset') == dataset)
    )

    # ticks = df.filter(filter)['SIM Layers'].unique().to_list()

    df = df.filter(filter).select(
        pl.col(
            'Accuracy Original Mimo',
            'Alignment Type',
        )
    )

    sns.barplot(
        df,
        y='Alignment Type',
        x='Accuracy Original Mimo',
        hue='Alignment Type',
        order=['Linear', 'Linear Precision', 'PPFE', 'Linear Entropy'],
        # markers=True,
        # dashes=True,
    )
    # Get all handles and labels
    # handles, labels = ax.get_legend_handles_labels()

    # alignment_labels = (
    #     df['Alignment Type']
    #     .sort(descending=True)
    #     .unique()
    #     .to_list()
    # )

    # alignment_handles = [handles[labels.index(cl)] for cl in alignment_labels]

    # legend1 = ax.legend(
    #     alignment_handles,
    #     alignment_labels,
    #     title='Alignment Type',
    #     ncol=2,
    #     loc='upper center',
    #     bbox_to_anchor=(0.7, 1.18),
    #     frameon=True,
    #     framealpha=1,
    # )

    # ax.add_artist(legend1)

    plt.ylabel('')
    plt.xlabel('Accuracy')
    # plt.xticks(ticks, labels=ticks)
    plt.xlim(0.8, 1.0)
    plt.savefig(
        str(img_path / f'AccuracyOriginalVsSIMLayers_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'AccuracyOriginalVsSIMLayers_{dataset}.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    return None


def taskfidelity_vs_simlayers(
    df: pl.DataFrame, dataset: str, img_path: Path
) -> None:
    """ """
    filter = (
        (pl.col('Simulation') == 'accuracyVSsimlayers')
        &
        # (pl.col('Weighted').is_null()) &
        (pl.col('Dataset') == dataset)
    )

    df = df.filter(filter).with_columns(
        (pl.col('Accuracy SIM Mimo') / pl.col('Accuracy Original Mimo')).alias(
            'Task Fidelity'
        )
    )

    ticks = df['SIM Layers'].unique().to_list()

    ax = sns.lineplot(
        df.drop('SIM Training Loss'),
        x='SIM Layers',
        y='Task Fidelity',
        hue='Intermediate Layers Atoms',
        style='Alignment Type',
        markers=True,
        dashes=True,
    )
    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    layer_labels = (
        df['Intermediate Layers Atoms']
        .sort(descending=False)
        .unique()
        .to_list()
    )

    alignment_labels = (
        df['Alignment Type'].sort(descending=False).unique().to_list()
    )

    layer_handles = [handles[labels.index(cl)] for cl in layer_labels]
    alignment_handles = [handles[labels.index(cl)] for cl in alignment_labels]

    legend1 = ax.legend(
        alignment_handles,
        alignment_labels,
        title='Alignment Type',
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.7, 1.18),
        frameon=True,
        framealpha=1,
    )

    ax.legend(
        layer_handles,
        layer_labels,
        title='Intermediate Layers Atoms',
        ncol=4,
        loc='upper center',
        bbox_to_anchor=(0.3, 1.18),
        frameon=True,
        framealpha=1,
    )

    ax.add_artist(legend1)

    plt.xlabel('SIM Layers')
    plt.ylabel('Task Fidelity')
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.savefig(
        str(img_path / f'TaskFidelityVsSIMLayers_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'TaskFidelityVsSIMLayers_{dataset}.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    return None


def mse_vs_simlayers(df: pl.DataFrame, dataset: str, img_path: Path) -> None:
    """"""
    filter = (
        (pl.col('Simulation') == 'accuracyVSsimlayers')
        &
        # (pl.col('Weighted').is_null()) &
        (pl.col('Dataset') == dataset)
    )

    ticks = df.filter(filter)['SIM Layers'].unique().to_list()

    sns.lineplot(
        df.filter(filter).drop('SIM Training Loss'),
        x='SIM Layers',
        y='MSE',
        hue='Intermediate Layers Atoms',
        style='MSE Type',
        markers=True,
        dashes=False,
    ).set(yscale='log')
    plt.xlabel('SIM Layers')
    plt.xticks(ticks, labels=ticks)
    plt.ylim(0, 1)
    plt.xlim(min(ticks), max(ticks))
    plt.savefig(
        str(img_path / f'MSEVsSIMLayers_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'MSEVsSIMLayers_{dataset}.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()
    return None


def sim_training_loss(df: pl.DataFrame, dataset: str, img_path: Path) -> None:
    """"""
    filter = (
        (pl.col('Simulation') == 'accuracyVSsimlayers')
        &
        # (pl.col('Weighted').is_null()) &
        (pl.col('Dataset') == dataset)
    )

    df_plot = (
        df.filter(filter)
        .select(
            pl.col(
                'SIM Layers',
                'SIM Training Loss',
                'Intermediate Layers Atoms',
                'Weighted',
                'Iterations',
                'Alignment Type',
            )
        )
        .with_columns(
            pl.int_ranges(1, pl.col('Iterations') + 1).alias('Iterations')
        )
        .explode('SIM Training Loss', 'Iterations')
        # .filter(pl.col('Iterations')<=500)
    )

    g = sns.relplot(
        df_plot.to_pandas(),
        x='Iterations',
        y='SIM Training Loss',
        col='Intermediate Layers Atoms',
        hue='SIM Layers',
        row='Alignment Type',
        kind='line',
        markers=True,
        dashes=False,
    ).set(xlabel='')
    g.set_titles('{col_name} | {row_name}')
    plt.savefig(
        str(img_path / f'TrainingLoss_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'TrainingLoss_{dataset}.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()
    return None


def accuracy_vs_lambda(df: pl.DataFrame, dataset: str, img_path: Path) -> None:
    """"""
    filter = (
        (pl.col('Simulation') == 'accuracyVSlambda')
        & (pl.col('Weighted').is_null())
        &
        # (pl.col('Lambda') > 0) &
        (pl.col('Dataset') == dataset)
        & (pl.col('Alignment Type') == 'Linear')
    )

    # ticks = df.filter(filter)['Lambda'].round(3).unique().to_list()
    ticks = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    sns.lineplot(
        df.filter(filter).to_pandas(),
        x='Lambda',
        y='Accuracy Sim',
        # row='Weighted',
        markers=True,
        dashes=False,
    ).set(xscale='log')
    # g.set_titles('{col_name} | Weighted {row_name}')
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylabel('Accuracy')
    plt.xlabel(r'$\lambda$')
    plt.savefig(
        str(img_path / f'AccuracyVSLambda_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'AccuracyVSLambda_{dataset}.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()
    return None


# =============================================================
#
#                     THE MAIN LOOP
#
# =============================================================


def main() -> None:
    """The main loop."""
    # Defining some usefull paths
    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results/'
    IMG_PATH: Path = CURRENT / 'img'

    # Create image Path
    IMG_PATH.mkdir(exist_ok=True)

    dataset = 'cifar10'

    # Set sns style
    sns.set_style('whitegrid')

    # Set style
    plt.style.use('.conf/plotting/plt.mplstyle')

    # Retrieve Data
    df: pl.DataFrame = (
        pl.read_parquet(RESULTS_PATH / 'linear/*.parquet')
        .with_columns(
            (
                (pl.col('SIM Meta Atoms Intermediate X')).cast(pl.String)
                + 'x'
                + (pl.col('SIM Meta Atoms Intermediate Y')).cast(pl.String)
            ).alias('Intermediate Layers Atoms')
        )
        .unpivot(
            cs.starts_with('MSE'),
            index=cs.all() - cs.starts_with('MSE'),
            variable_name='MSE Type',
            value_name='MSE',
        )
        .with_columns(pl.col('Weighted').cast(pl.String))
        .sort(
            ['SIM Meta Atoms Intermediate X', 'Alignment Type'],
            descending=False,
        )
    )

    # ===================================================================================
    #                          Accuracy Vs SIM Layers
    # ===================================================================================
    accuracy_vs_simlayers(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          Accuracy Vs SNR
    # ===================================================================================
    accuracy_vs_snr(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          Accuracy Original Vs SIM Layers
    # ===================================================================================
    accuracy_original_vs_simlayers(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          Task Fidelity Vs SIM Layers
    # ===================================================================================
    taskfidelity_vs_simlayers(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          MSE Loss Vs SIM Layers
    # ===================================================================================
    # mse_vs_simlayers(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          Accuracy Vs Lambda
    # ===================================================================================
    # accuracy_vs_lambda(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          SIM Training Loss Vs SIM Layers
    # ===================================================================================
    # sim_training_loss(df, dataset, IMG_PATH)

    return None


if __name__ == '__main__':
    main()
