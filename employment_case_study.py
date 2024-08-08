from pathlib import Path

import folktables as ft
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics as skm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import common as cmn
from binary_confusion_matrix import BinaryConfusionMatrix as BCM
from common import AlgorithmData

#available datasets from ACSDataSource
ACS_DATASETS = (
    ft.ACSEmployment,  #is employed is positive class. Rac1P protected group.
    ft.ACSHealthInsurance,
    ft.ACSPublicCoverage,
    ft.ACSTravelTime,
    ft.ACSMobility,
    ft.ACSEmploymentFiltered,
    ft.ACSIncomePovertyRatio
)

STATE_LIST = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

AVAILABLE_YEARS = [str(year) for year in range(2014, 2019)]

DATA_ROOT_DIR = Path("datasets/folktables/data")

###Important link for this data
### https://www.census.gov/programs-surveys/acs/microdata/documentation.2018.html#list-tab-QLNP5P9TEQCUN4SN46

ORDINAL_MAP = {
    "RAC1P": {1: "White", 2: "Black", 3: "American Indian", 4: "Alaska Native",
              5: "American Indian and/or Alaska Native", 6: "Asian", 7: "Native Hawaiian and other Pacific Islander",
              8: "Some other race alone", 9: "Two or more races"}
}


class FolkAlgorithmData(AlgorithmData):
    def __init__(self, acs_problem: ft.BasicProblem):
        super().__init__(
            train_frac=.6,
            feat_names=acs_problem.features,
            group_name=acs_problem.group,
            label_names=[acs_problem.target],
            ordinal_map=ORDINAL_MAP
        )
        self.problem = acs_problem

    def data_from_acs_data(self, acs_data):
        features, label, group = self.problem.df_to_numpy(acs_data)
        assert self.problem.group in self.problem.features
        self._data = pd.DataFrame(features, columns=self.feat_names)
        self._data[self.label_names] = pd.DataFrame(label.reshape((-1, 1)), columns=self.label_names)
        self._data = self._data.astype(float)
        return


def get_data(acs_problem=ft.ACSEmployment, states=["GA"], years=['2014'], survey='person'):
    data_years = []
    for year in years:
        data_source = ft.ACSDataSource(survey_year=year, horizon='1-Year', survey=survey)
        acs_data = data_source.get_data(states=states, download=True)

        data = FolkAlgorithmData(acs_problem)
        data.data_from_acs_data(acs_data)
        data.normalize()
        data.shuffle()
        data_years.append(data)

    return data_years


def plot_heatmaps(data: FolkAlgorithmData, clf, suptitle:str):

    clf.fit(data.trainx, data.trainy)
    cms = {}
    stats = {}

    for key, group in data.test.groupby(data.group_name):
        group_nm = data.get_ordinal_name(key, data.group_name)
        cms[group_nm] = skm.confusion_matrix(group[data.label_names], clf.predict(group[data.feat_names]),
                                             labels=[0, 1])
        cms[group_nm] = BCM(cms[group_nm])
        stats[group_nm] = {
            "TE": cms[group_nm].treatment_equality_part().item(),
            "DI": cms[group_nm].disparate_impact_part().item(),
            "OFI": cms[group_nm].marginal_benefit().item(),
            "PP": cms[group_nm].predictive_parity_part().item()
        }

    # print("Confusion Matrices:")
    # print(''.join([f"{key}:\n{cm}\n" for key, cm in cms.items()]))

    stats = pd.DataFrame.from_dict(stats)
    stats.rename(columns={
        "American Indian": "American Indian (AI)",
        "Alaska Native": "Alaska Native (AN)",
        "American Indian and/or Alaska Native": "AI and/or AN",
        "Native Hawaiian and other Pacific Islander": "Pacific Islander",
        "Some other": "Some other",
    }, inplace=True)

    race_pairs = {}
    for race1 in stats.columns:
        for race2 in stats.columns:
            te = stats[race1]["TE"] - stats[race2]["TE"]
            di = stats[race1]["DI"] / stats[race2]["DI"]
            ofi = stats[race1]["OFI"] - stats[race2]["OFI"]
            pp = stats[race1]["PP"] - stats[race2]["PP"]
            race_pairs[(race1, race2)] = {"TE": te, "DI": di, "OFI": ofi, "PP": pp}

    race_pairs = pd.DataFrame(race_pairs).T.reset_index(names=['Race i', 'Race j'])
    # Replace inf with nan for the color map to work properly
    race_pairs.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)
    # Prepare the data for heatmap
    pivot_TE = race_pairs.pivot(index="Race i", columns="Race j", values="TE")
    pivot_PP = race_pairs.pivot(index="Race i", columns="Race j", values="PP")
    pivot_DI = race_pairs.pivot(index="Race i", columns="Race j", values="DI")
    pivot_ofi = race_pairs.pivot(index="Race i", columns="Race j", values="OFI")

    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(15, 13.5))
    ax = ax.flatten()

    # heat_color_map = mcolors.LinearSegmentedColormap.from_list(
    #     "mycmap", ["royalblue", "whitesmoke", "firebrick"])
    heat_color_map = "coolwarm"
    axis_title_size = 14

    sns.heatmap(pivot_TE, annot=True, cmap="coolwarm_r", ax=ax[0], fmt=".3f")
    ax[0].set_title('TE Heatmap', fontsize=axis_title_size)

    sns.heatmap(pivot_PP, annot=True, cmap=heat_color_map, ax=ax[1], fmt=".3f")
    ax[1].set_title('PP Heatmap', fontsize=axis_title_size)

    sns.heatmap(pivot_DI, annot=True, cmap=heat_color_map, ax=ax[2], fmt=".3f")
    ax[2].set_title('DI Heatmap', fontsize=axis_title_size)

    sns.heatmap(pivot_ofi, annot=True, cmap=heat_color_map, ax=ax[3], fmt=".3f")
    ax[3].set_title('OFI Heatmap', fontsize=axis_title_size)

    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=90, fontsize=12)
        a.set_yticklabels(a.get_yticklabels(), rotation=0, fontsize=12)
        current_xlabel = a.get_xlabel()
        a.set_xlabel(current_xlabel, fontsize=13)
        current_ylabel = a.get_ylabel()
        a.set_ylabel(current_ylabel, fontsize=13)

    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    return


if __name__ == "__main__":
    np.random.seed(0)

    data = get_data(ft.ACSEmployment, ["GA"], years=['2014', '2015', '2016', '2017'], )
    # Shape of (393236, 17)
    data_df = pd.concat([d.data for d in data])
    print(f"Employment records count: {data_df.shape[0]:,}")
    data = data[0]

    data._train_frac = .6  # This is default, but just to help readers' understandability
    # we set it here again.

    # Try 60 random samples
    data._data = data_df.sample(n=60)

    import warnings

    warnings.filterwarnings("ignore")
    plot_heatmaps(data, clf=RandomForestClassifier(), suptitle="Folktables Employment: Random Forest with Few Samples")
    plt.savefig(cmn.RESULTS_DIR / "folktables_employment_ofi_nuance.png")

    # Use all data
    data._data = data_df

    plot_heatmaps(data, clf=RandomForestClassifier(), suptitle="Folktables Employment: Random Forest with All Samples")
    plt.savefig(cmn.RESULTS_DIR / "folktables_employment_GA_2014_to_2017.png")
    plot_heatmaps(data, clf=GaussianNB(), suptitle="Folktables Employment: Naive Bayes with All Samples")
    plt.savefig(cmn.RESULTS_DIR / "folktables_employment_GA_2014_to_2017_NB.png")

    plt.show()
    pass
