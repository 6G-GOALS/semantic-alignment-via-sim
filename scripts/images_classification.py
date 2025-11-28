"""A usefull script used to compute the needed plots.

The script expects to have the results saved in the following structure (both linear and baseline results):
|_ results/
    |_ classification/
       |_ r1.parquet
       |_ ...
       |_ rk.parquet
"""

import polars as pl
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt


def accuracy_vs_simlayers(
    df: pl.DataFrame, dataset: str, img_path: Path
) -> None:
    """ """
    filter = (
        (pl.col('Simulation') == 'accuracyVSsimlayers')
        & (pl.col('Alignment Type').is_in({'PPFE', 'Linear'}))
        & (pl.col('Dataset') == dataset)
    )

    df = df.filter(filter)

    ticks = df['SIM Layers'].unique().to_list()

    original_acc = (
        df.select('Accuracy No Mismatch')
        .unique('Accuracy No Mismatch')
        .mean()
        .item()
    )

    acc_original_no_mimo = dict(
        df.group_by('Alignment Type')
        .agg(pl.col('Accuracy Original No Mimo').mean())
        .rows()
    )

    ax = sns.lineplot(
        df.drop('SIM Training Loss'),
        x='SIM Layers',
        y='Accuracy SIM Mimo',
        hue='Intermediate Layers Atoms',
        style='Alignment Type',
        markers=True,
        dashes=True,
    )

    line1 = plt.axhline(
        y=original_acc,
        color='gray',
        linestyle='-',
        label='No Mismatch',
        linewidth=2,
    )

    line2 = plt.axhline(
        y=acc_original_no_mimo['Linear'],
        color='gray',
        linestyle='-.',
        linewidth=2,
        label='Original Linear',
    )

    line3 = plt.axhline(
        y=acc_original_no_mimo['PPFE'],
        color='gray',
        linestyle=':',
        linewidth=2,
        label='Original PPFE',
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
        bbox_to_anchor=(0.37, 0.18),
        frameon=True,
        framealpha=1,
    )

    legend2 = ax.legend(
        layer_handles,
        layer_labels,
        title='Intermediate Layers Atoms',
        ncol=4,
        loc='upper center',
        bbox_to_anchor=(0.76, 0.18),
        frameon=True,
        framealpha=1,
    )

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.legend(
        handles=[
            line1,
            line2,
            line3,
        ],
        title='',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=True,
        framealpha=1,
        borderpad=0.5,
    )

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


def accuracy_vs_thickness(
    df: pl.DataFrame, dataset: str, img_path: Path
) -> None:
    """ """
    filter = (pl.col('Simulation') == 'accuracyVSthickness') & (
        pl.col('Dataset') == dataset
    )

    amplification = 1.5

    df = (
        df.filter(filter)
        .with_columns(
            pl.when(
                (pl.col('SIM Spacing Divisor Input') == 2)
                & (pl.col('SIM Spacing Divisor Output') == 2)
                & (pl.col('SIM Spacing Divisor Intermediate') == 2)
            )
            .then(pl.lit('No Amplification'))
            .when(
                (pl.col('SIM Spacing Divisor Input') == amplification)
                & (pl.col('SIM Spacing Divisor Output') == 2)
                & (pl.col('SIM Spacing Divisor Intermediate') == 2)
            )
            .then(pl.lit('Input Layer'))
            .when(
                (pl.col('SIM Spacing Divisor Input') == amplification)
                & (pl.col('SIM Spacing Divisor Output') == 2)
                & (pl.col('SIM Spacing Divisor Intermediate') == amplification)
            )
            .then(pl.lit('Input & Intermediate Layers'))
            .when(
                (pl.col('SIM Spacing Divisor Input') == amplification)
                & (pl.col('SIM Spacing Divisor Output') == amplification)
                & (pl.col('SIM Spacing Divisor Intermediate') == 2)
            )
            .then(pl.lit('Input & Output Layers'))
            .when(
                (pl.col('SIM Spacing Divisor Input') == amplification)
                & (pl.col('SIM Spacing Divisor Output') == amplification)
                & (pl.col('SIM Spacing Divisor Intermediate') == amplification)
            )
            .then(pl.lit('All Layers'))
            .otherwise(None)
            .alias('Amplification')
        )
        .filter(
            pl.col('Amplification').is_in({'Input Layer', 'No Amplification'})
        )
        .drop_nulls('Amplification')
        .sort('Amplification', descending=True)
        .with_columns(
            (pl.col('Thickness Multiplier') * pl.col('SIM Wavelength')).alias(
                'SIM Layer Distance'
            )
        )
    )

    ticks = df['SIM Layer Distance'].unique().to_list()

    original_acc = (
        df.select('Accuracy No Mismatch')
        .unique('Accuracy No Mismatch')
        .mean()
        .item()
    )
    acc_original_no_mimo = dict(
        df.group_by('Alignment Type')
        .agg(pl.col('Accuracy Original No Mimo').mean())
        .rows()
    )

    ax = sns.lineplot(
        df.drop('SIM Training Loss'),
        x='SIM Layer Distance',
        y='Accuracy SIM Mimo',
        hue='Intermediate Layers Atoms',
        style='Amplification',
        markers=True,
        dashes=True,
    )

    line1 = plt.axhline(
        y=original_acc,
        color='gray',
        linestyle='-',
        label='No Mismatch',
        linewidth=2,
    )

    line2 = plt.axhline(
        y=acc_original_no_mimo['PPFE'],
        color='gray',
        linestyle=':',
        label='Original PPFE',
        linewidth=2,
    )

    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    layer_labels = (
        df.filter(filter)['Intermediate Layers Atoms']
        .sort(descending=True)
        .unique()
        .to_list()
    )

    amplification_labels = (
        df.filter(filter)['Amplification']
        .sort(descending=True)
        .unique()
        .to_list()
    )

    layer_labels = sorted(layer_labels)
    amplification_labels = sorted(amplification_labels)

    layer_handles = [handles[labels.index(cl)] for cl in layer_labels]
    amplification_handles = [
        handles[labels.index(cl)] for cl in amplification_labels
    ]

    legend1 = ax.legend(
        amplification_handles,
        amplification_labels,
        title='Amplification',
        ncol=2,
        loc='upper center',
        # bbox_to_anchor=(0.63, 0.18),
        bbox_to_anchor=(0.23, 0.18),
        frameon=True,
        framealpha=1,
    )

    legend2 = ax.legend(
        layer_handles,
        layer_labels,
        title='Intermediate Layers Atoms',
        ncol=4,
        loc='upper center',
        # bbox_to_anchor=(0.24, 0.18),
        bbox_to_anchor=(0.61, 0.18),
        frameon=True,
        framealpha=1,
    )

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.legend(
        handles=[
            line1,
            line2,
        ],
        title='',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=True,
        framealpha=1,
        borderpad=0.5,
    )

    plt.xlabel(r'$s_{\mathrm{layer}}$')
    plt.ylabel('Accuracy')
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    plt.savefig(
        str(img_path / f'AccuracyVsThickness_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'AccuracyVsThickness_{dataset}.png'),
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

    df = df.filter(filter)

    ticks = df['SNR [dB]'].unique().to_list()

    original_acc = (
        df.select('Accuracy No Mismatch')
        .unique('Accuracy No Mismatch')
        .mean()
        .item()
    )

    original_acc_no_mimo = (
        df.select('Accuracy Original No Mimo')
        .unique('Accuracy Original No Mimo')
        .mean()
        .item()
    )

    alignment_type = (
        df.select('Alignment Type').unique('Alignment Type').item()
    )

    ax = sns.lineplot(
        df.drop('SIM Training Loss'),
        x='SNR [dB]',
        y='Accuracy SIM Mimo',
        hue='Intermediate Layers Atoms',
        style='SIM Layers',
        markers=True,
        dashes=True,
    )

    line1 = plt.axhline(
        y=original_acc,
        color='gray',
        linestyle='-',
        label='No Mismatch',
        linewidth=2,
    )

    line2 = plt.axhline(
        y=original_acc_no_mimo,
        color='gray',
        linestyle=':',
        label='Original ' + alignment_type,
        linewidth=2,
    )

    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    layer_labels = (
        df.filter(filter)['Intermediate Layers Atoms'].unique().to_list()
    )

    num_layers_labels = df.filter(filter)['SIM Layers'].unique().to_list()

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
        # bbox_to_anchor=(0.57, 0.18),
        bbox_to_anchor=(0.41, 0.18),
        frameon=True,
        framealpha=1,
    )

    legend2 = ax.legend(
        layer_handles,
        layer_labels,
        title='Intermediate Layers Atoms',
        ncol=4,
        loc='upper center',
        # bbox_to_anchor=(0.84, 0.18),
        bbox_to_anchor=(0.76, 0.18),
        frameon=True,
        framealpha=1,
    )

    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.legend(
        handles=[
            line1,
            line2,
        ],
        title='',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=True,
        framealpha=1,
        borderpad=0.5,
    )

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


def accuracy_original(df: pl.DataFrame, dataset: str, img_path: Path) -> None:
    """ """
    filter = (
        (pl.col('Simulation') == 'accuracyVSsimlayers')
        &
        # (pl.col('Weighted').is_null()) &
        (pl.col('Dataset') == dataset)
    )

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
        order=['Linear', 'Procrustes', 'PPFE'],
    )

    plt.ylabel('')
    plt.xlabel('Accuracy')
    plt.xlim(0.8, 1.0)
    plt.savefig(
        str(img_path / f'AccuracyOriginal_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'AccuracyOriginal_{dataset}.png'),
        bbox_inches='tight',
    )
    plt.clf()
    plt.cla()

    return None


def taskfidelity_vs_simlayers(
    df: pl.DataFrame, dataset: str, img_path: Path
) -> None:
    """ """
    filter = (pl.col('Simulation') == 'accuracyVSsimlayers') & (
        pl.col('Dataset') == dataset
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


def sim_training_loss(df: pl.DataFrame, dataset: str, img_path: Path) -> None:
    """"""
    filter = (pl.col('Simulation') == 'accuracyVSsimlayers') & (
        pl.col('Dataset') == dataset
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


def sim_training_loss_2(
    df: pl.DataFrame, dataset: str, img_path: Path
) -> None:
    """"""
    filter = (
        (pl.col('Simulation') == 'accuracyVSsimlayers')
        & (pl.col('Dataset') == dataset)
        & (pl.col('Intermediate Layers Atoms') == '64x64')
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
    )

    g = sns.relplot(
        df_plot.to_pandas(),
        x='Iterations',
        y='SIM Training Loss',
        hue='SIM Layers',
        col='Alignment Type',
        kind='line',
        markers=True,
        dashes=False,
    ).set(xlabel='')
    g.set_titles('{col_name}')
    plt.savefig(
        str(img_path / f'TrainingLoss_only64x64_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'TrainingLoss_only64x64_{dataset}.png'),
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
        & (pl.col('Dataset') == dataset)
        & (pl.col('Alignment Type') == 'Linear')
    )

    ticks = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    sns.lineplot(
        df.filter(filter).to_pandas(),
        x='Lambda',
        y='Accuracy Sim',
        # row='Weighted',
        markers=True,
        dashes=False,
    ).set(xscale='log')
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


def accuracy_vs_ppfe(df: pl.DataFrame, dataset: str, img_path: Path) -> None:
    """ """
    filter = (pl.col('Simulation') == 'ppfe') & (pl.col('Dataset') == dataset)

    ticks = df.filter(filter)['Number Clusters'].unique().to_list()

    df = (
        df.drop('SIM Training Loss')
        .filter(filter)
        .rename(
            {
                'Accuracy SIM Mimo': 'SIM',
                'Accuracy Original Mimo': 'Original',
            }
        )
        .unpivot(
            on=['SIM', 'Original'],
            index=['Number Clusters', 'Number Proto'],
            variable_name='Type',
            value_name='Accuracy',
        )
    )

    sns.lineplot(
        df,
        x='Number Clusters',
        y='Accuracy',
        hue='Type',
        style='Number Proto',
        markers=True,
        dashes=True,
    ).set(xscale='log')

    plt.xlabel('Proto Anchors')
    plt.ylabel('Accuracy')
    plt.xticks(ticks, labels=ticks)
    plt.xlim(min(ticks), max(ticks))
    plt.ylim(None, 1.0)
    plt.savefig(
        str(img_path / f'AccuracyVsPPFE_{dataset}.pdf'),
        format='pdf',
        bbox_inches='tight',
    )
    plt.savefig(
        str(img_path / f'AccuracyVsPPFE_{dataset}.png'),
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
    IMG_PATH: Path = CURRENT / 'img/classification'

    # Create image Path
    IMG_PATH.mkdir(exist_ok=True)

    dataset = 'cifar10'

    # Set sns style
    sns.set_style('whitegrid')

    # Set style
    plt.style.use('.conf/plotting/plt.mplstyle')

    # Retrieve Data
    df: pl.DataFrame = (
        pl.read_parquet(RESULTS_PATH / 'classification/*.parquet')
        .with_columns(
            (
                (pl.col('SIM Meta Atoms Intermediate X')).cast(pl.String)
                + 'x'
                + (pl.col('SIM Meta Atoms Intermediate Y')).cast(pl.String)
            ).alias('Intermediate Layers Atoms')
        )
        .with_columns(pl.col('Weighted').cast(pl.String))
        .sort(
            ['SIM Meta Atoms Intermediate X', 'Alignment Type'],
            descending=False,
        )
        .with_columns(
            (
                pl.col('SIM Thickness')
                / (pl.col('SIM Layers') * pl.col('SIM Wavelength'))
            )
            .round()
            .cast(pl.UInt16)
            .alias('Thickness Multiplier')
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
    # accuracy_original(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          Task Fidelity Vs SIM Layers
    # ===================================================================================
    # taskfidelity_vs_simlayers(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          Accuracy Vs PPFE Clusters
    # ===================================================================================
    # accuracy_vs_ppfe(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          Accuracy Vs Thickness
    # ===================================================================================
    accuracy_vs_thickness(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          Accuracy Vs Lambda
    # ===================================================================================
    # accuracy_vs_lambda(df, dataset, IMG_PATH)

    # ===================================================================================
    #                          SIM Training Loss Vs SIM Layers
    # ===================================================================================
    # sim_training_loss(df, dataset, IMG_PATH)
    # sim_training_loss_2(df, dataset, IMG_PATH)

    return None


if __name__ == '__main__':
    main()
