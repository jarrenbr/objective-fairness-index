import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from binary_confusion_matrix import BinaryConfusionMatrix
import common as cmn


def compas_ethnicity():
    nAfricanAmer = 1918
    nAsian = 26
    nWhite = 1459
    nHispanic = 355
    nNativeAmerican = 7
    nOther = 255

    # Creating a dictionary to hold the Binary Confusion Matrices (BCM) for each race/ethnic group
    # Using Table 8B: https://ar5iv.labs.arxiv.org/html/2307.00472v1

    confusion_matrices = {
        "African Amer.": BinaryConfusionMatrix([[13, 8], [24, 55]]) / 100 * nAfricanAmer,
        "Asian": BinaryConfusionMatrix([[12, 0], [4, 85]]) / 100 * nAsian,
        "Caucasian": BinaryConfusionMatrix([[4, 8], [14, 75]]) / 100 * nWhite,
        "Hispanic": BinaryConfusionMatrix([[3, 7], [17, 73]]) / 100 * nHispanic,
        "Native Amer.": BinaryConfusionMatrix([[14, 0], [14, 71]]) / 100 * nNativeAmerican,
        "Other": BinaryConfusionMatrix([[7, 7], [13, 74]]) / 100 * nOther
    }

    # We assume the positive prediction is beneficial. Originally, the positive prediction was recidivism.
    for key in confusion_matrices.keys():
        confusion_matrices[key].flip_pos_neg_labels()

    stats = {}
    for ethn, cm in confusion_matrices.items():
        stats[ethn] = {
            "TE": cm.treatment_equality_part().item(),
            "DI": cm.disparate_impact_part().item(),
            "OFI": cm.marginal_benefit().item(),
            "PP": cm.predictive_parity_part().item()
        }

    stats = pd.DataFrame.from_dict(stats)

    ethn_pairs = {}
    import warnings
    warnings.filterwarnings("ignore")
    for ethn1 in stats.columns:
        for ethn2 in stats.columns:
            te = stats[ethn1]["TE"] - stats[ethn2]["TE"]
            di = stats[ethn1]["DI"] / stats[ethn2]["DI"]
            ofi = stats[ethn1]["OFI"] - stats[ethn2]["OFI"]
            pp = stats[ethn1]["PP"] - stats[ethn2]["PP"]
            ethn_pairs[(ethn1, ethn2)] = {"TE": te, "DI": di, "OFI": ofi, "PP": pp}
    warnings.filterwarnings("default")

    ethn_pairs = pd.DataFrame(ethn_pairs).T.reset_index(names=['Ethnicity i', 'Ethnicity j'])
    # Prepare the data for heatmap
    pivot_TE = ethn_pairs.pivot(index="Ethnicity i", columns="Ethnicity j", values="TE")
    pivot_DI = ethn_pairs.pivot(index="Ethnicity i", columns="Ethnicity j", values="DI")
    pivot_ofi = ethn_pairs.pivot(index="Ethnicity i", columns="Ethnicity j", values="OFI")
    pivot_PP = ethn_pairs.pivot(index="Ethnicity i", columns="Ethnicity j", values="PP")

    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()
    axis_title_size = 14

    sns.heatmap(pivot_TE, annot=True, cmap="coolwarm_r", ax=ax[0])
    ax[0].set_title('TE Heatmap', fontsize=axis_title_size)

    sns.heatmap(pivot_PP, annot=True, cmap="coolwarm", ax=ax[1])
    ax[1].set_title('PP Heatmap', fontsize=axis_title_size)

    sns.heatmap(pivot_DI, annot=True, cmap="coolwarm", ax=ax[2])
    ax[2].set_title('DI Heatmap', fontsize=axis_title_size)

    sns.heatmap(pivot_ofi, annot=True, cmap="coolwarm", ax=ax[3])
    ax[3].set_title('OFI Heatmap', fontsize=axis_title_size)

    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=90, fontsize=12)
        a.set_yticklabels(a.get_yticklabels(), rotation=0, fontsize=12)
        current_xlabel = a.get_xlabel()
        a.set_xlabel(current_xlabel, fontsize=13)
        current_ylabel = a.get_ylabel()
        a.set_ylabel(current_ylabel, fontsize=13)

    plt.suptitle("COMPAS Heatmaps", fontsize=16)
    plt.tight_layout()
    plt.savefig(cmn.RESULTS_DIR / "compas_ethnicity.png")
    plt.show()

    return


if __name__ == "__main__":
    compas_ethnicity()
    pass
