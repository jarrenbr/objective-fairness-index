from pathlib import Path
from abc import ABC

from sklearn import preprocessing as skpp
import pandas as pd
import numpy as np

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class Data(ABC):

    def __init__(self, feat_names, group_name, label_names, ordinal_map, data=None, scaler=None):
        self._label_names = label_names
        self._group_name = group_name
        self._feat_names = feat_names
        self._data = data
        self._ordinal_map = ordinal_map
        self.scaler = scaler

    def get_ordinal_name(self, val: float, colname: str) -> str:
        ordinal = round(self.revert_value(val, colname))
        name = self.ordinal_map[colname][ordinal]
        return name

    def normalize(self):
        assert self.scaler is None
        self.scaler = skpp.MinMaxScaler()
        self._data = pd.DataFrame(
            self.scaler.fit_transform(self._data),
            columns=self.data.columns
        )

    def revert_value(self, val, colname):
        position = np.where(self.scaler.feature_names_in_ == colname)[0][0]
        arr = np.zeros((1, self.scaler.n_features_in_))
        arr[..., position] = val
        inv_trans = self.scaler.inverse_transform(arr)
        return inv_trans[..., position].flatten()[0]

    def shuffle(self):
        self._data = self._data.sample(frac=1)

    def split_groups(self, colname):
        grouped = self.data.groupby(colname)
        for key, group in grouped:
            ordinal_name = self.get_ordinal_name(key, colname)
            yield ordinal_name, group[self._feat_names], group[self._label_names]

    def over_groups(self, colname, func_x_y):
        results = {}
        for group, x, y in self.split_groups(colname):
            results[group] = func_x_y(x, y)
        return pd.Series(results)

    def over_each_group_combination(self, colname, func_of_xys, reduce=None):
        """
        :param colname:
        :param func_of_xys: func(xy1:XY, xy2:XY)
        :param reduce: reduces along axis 0. options: None, mean
        :return:
        """
        results = {}
        for groupi, xi, yi in self.split_groups(colname):
            datai = XY(xi, yi)
            results[groupi] = {}
            for groupj, xj, yj in self.split_groups(colname):
                results[groupi][groupj] = func_of_xys(datai, XY(xj, yj))

        results = pd.DataFrame(results)
        if reduce == "mean":
            results = results.mean()

        return results

    @property
    def data(self):
        return self._data

    @property
    def x(self):
        return self.data[self.feat_names]

    @property
    def y(self):
        return self.data[self.label_names]

    @property
    def label_names(self):
        return self._label_names

    @property
    def group_name(self):
        return self._group_name

    @property
    def feat_names(self):
        return self._feat_names

    @property
    def ordinal_map(self):
        return self._ordinal_map


class AlgorithmData(Data):
    def __init__(self, train_frac, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_frac = train_frac

    def _get_pivot(self):
        return round(self.data.shape[0] * self._train_frac)

    @property
    def train_frac(self):
        return self._train_frac

    @property
    def trainx(self):
        return self.x[:self._get_pivot()]

    @property
    def trainy(self):
        return self.y[:self._get_pivot()]

    @property
    def testx(self):
        return self.x[self._get_pivot():]

    @property
    def testy(self):
        return self.y[self._get_pivot():]

    @property
    def test(self):
        return self.data[self._get_pivot():]

    @property
    def train(self):
        return self.data[:self._get_pivot()]


class XY:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


class MlData:
    def __init__(self, train=None, validate=None, test=None):
        self.train = train
        self.test = test
        self.validate = validate

    _colToOrd = {col: i for i, col in enumerate(("train", "validate", "test"))}

    def __len__(self):
        return (self.train is not None) + (self.test is not None) + (self.validate is not None)

    def __getitem__(self, item):
        # return ord(item)
        if isinstance(item, str):
            item = MlData._colToOrd[item]
        if item == 0:
            return self.train
        elif item == 1:
            return self.validate
        elif item == 2:
            return self.test
        raise "Invalid item."

    def apply(self, function, *args, **kwargs):
        mlKwargs = {}
        for key, value in {"train": self.train, "test": self.test, "validate": self.validate}.items():
            if value is not None:
                mlKwargs[key] = function(value, *args, **kwargs)

        return MlData(**mlKwargs)

    def transform(self, function, *args, **kwargs):
        if self.train is not None:
            self.train = function(self.train, *args, **kwargs)
        if self.test is not None:
            self.test = function(self.test, *args, **kwargs)
        if self.validate is not None:
            self.validate = function(self.validate, *args, **kwargs)

    def __iter__(self):
        yield self.train
        yield self.test
        yield self.validate
