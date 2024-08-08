import numpy as np

"""
LEGEND
pp = predicted positive
p = positive label
pn = predicted negative
n = negative label
tp, fp, fn, tn = confusion matrix cells
cm = confusion matrix (np.ndarray)
"""
"""CM Format
[[tp, fn],
 [fp, tn]]
"""


class BinaryConfusionMatrix(np.ndarray):
    def __new__(cls, input_array: np.ndarray | list | dict, normalize=False, dtype=np.float32):
        if isinstance(input_array, dict):
            obj = np.array([[input_array["TP"], input_array["FN"]], [input_array["FP"], input_array["TN"]]])
        else:
            obj = np.asarray(input_array, dtype=dtype).view(cls)
        if obj.shape[-2:] != (2, 2):
            raise ValueError("Array shape must be (Any, 2,2)")
        # if obj.ndim == 2:
        #     obj = obj.reshape((-1, 2, 2))
        obj.counts = obj.copy()
        if normalize:
            obj.normalize()
        return obj

    def __array_finalize__(self, obj):
        if obj is None or hasattr(self, '_is_normalized'):
            return
        self._is_normalized = False

    @property
    def tp(self):
        return self[..., 0, 0]

    @property
    def fn(self):
        return self[..., 0, 1]

    @property
    def fp(self):
        return self[..., 1, 0]

    @property
    def tn(self):
        return self[..., 1, 1]

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized

    def flip_pos_neg_labels(self):
        tmp = self[..., 0, 1].copy()
        self[..., 0, 1] = self[..., 1, 0]
        self[..., 1, 0] = tmp

        tmp = self[..., 0, 0].copy()
        self[..., 0, 0] = self[..., 1, 1]
        self[..., 1, 1] = tmp

    def normalize(self):
        self._is_normalized = True
        self /= np.sum(self, axis=(-1, -2), keepdims=True)

    def pp(self):
        """Predicted positive"""
        return self.tp + self.fp

    def pn(self):
        """Predicted negative"""
        return self.fn + self.tn

    def p(self):
        """Labelled positive"""
        return self.tp + self.fn

    def n(self):
        """Labelled negative"""
        return self.fp + self.tn

    def __str__(self):
        return f"{self.tp}, {self.fn}\n{self.fp}, {self.tn}"

    # bounded
    def accuracy(self):
        if self.is_normalized:
            return self.tp + self.tn
        else:
            return (self.tp + self.tn) / self.sum(axis=(-1, -2), keepdims=True)

    def mcc(self):
        """phi coefficient (φ or r_φ) or Matthews correlation coefficient (MCC)"""
        numerator = self.tp * self.tn - self.fp * self.fn
        denominator = np.sqrt(self.pp() * self.p() * self.n() * self.pn())
        return numerator / denominator

    # unbounded
    def ppv(self):
        """precision or positive predictive value (PPV)"""
        return self.tp / self.pp()

    def npv(self):
        """negative predictive value (NPV)"""
        return self.tn / self.pn()

    def tpr(self):
        """sensitivity, recall, hit rate, or true positive rate (TPR)"""
        return self.tp / self.p()

    def fpr(self):
        """fall-out or false positive rate (FPR)"""
        return self.fp / self.n()

    def tnr(self):
        """specificity, selectivity or true negative rate (TNR)"""
        return self.tn / self.n()

    def fnr(self):
        """miss rate or false negative rate (FNR)"""
        return self.fn / self.p()

    def fdr(self):
        """false discovery rate (FDR)"""
        return self.fp / self.pp()

    def for_(self):
        """false omission rate (FOR)"""
        return self.fn / self.pn()

    def equalized_odds_part(self):
        return self.tpr() + self.fpr()

    def treatment_equality_part(self):
        return self.fn / self.fp

    def disparate_impact_part(self):
        return self.pp() / self.p()

    def predictive_parity_part(self):
        return self.tp / self.p()

    def marginal_benefit(self):
        if self.is_normalized:
            return self.fp - self.fn
        else:
            return (self.fp - self.fn) / self.sum(axis=(-1, -2), keepdims=True)


name_funcs = {
    func.__name__.title(): func
    for func in
    [BinaryConfusionMatrix.accuracy, BinaryConfusionMatrix.mcc, BinaryConfusionMatrix.equalized_odds_part,
     BinaryConfusionMatrix.treatment_equality_part, BinaryConfusionMatrix.disparate_impact_part,
     BinaryConfusionMatrix.predictive_parity_part, BinaryConfusionMatrix.ppv,
     BinaryConfusionMatrix.npv, BinaryConfusionMatrix.tpr, BinaryConfusionMatrix.fpr,
     BinaryConfusionMatrix.tnr, BinaryConfusionMatrix.fnr, BinaryConfusionMatrix.fdr,
     BinaryConfusionMatrix.for_]
}

def objective_fairness_index(cm1: BinaryConfusionMatrix, cm2: BinaryConfusionMatrix) -> float:
    return (cm1.marginal_benefit() - cm2.marginal_benefit()).item()

class AdHocBinaryCm:
    """Ad hoc binary confusion matrix for easy updating of cells to build a cm."""

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def array(self) -> np.ndarray:
        return np.asarray([
            [self.tp, self.fn],
            [self.fp, self.tn]
        ])

    def to_binary_cm(self, **kwargs):
        return BinaryConfusionMatrix(self.array(), **kwargs)


def print_scenario(cm1: BinaryConfusionMatrix, cm2: BinaryConfusionMatrix):
    print(f"CM1:\n{cm1}\nCM2:\n{cm2}")
    np.seterr(divide='ignore', invalid='ignore')
    te = cm1.treatment_equality_part() - cm2.treatment_equality_part()
    di = cm1.disparate_impact_part() / cm2.disparate_impact_part()
    ofi = objective_fairness_index(cm1, cm2)
    np.seterr(divide='warn', invalid='warn')
    print(f"TE: {te.item()}, DI: {di.item()}, OFI: {ofi}")
    return te, di, ofi


if __name__ == "__main__":
    BCM = BinaryConfusionMatrix

    ci = BCM([[1000,1], [20,0]])
    cj = BCM([[1000,20], [1,0]])

    di = BCM([[0,1], [20,0]])

    print("""
    Recall that a negative value indicates bias against i (i's (FP-FN)/n > j's (FP-FN)/n),\
     and a positive value indicates bias against j.
    """)

    BCM = BinaryConfusionMatrix
    ai = BCM([[1, 0], [0, 5]])
    aj = BCM([[7, 0], [1, 10]])
    print("\nScenario a")
    print_scenario(ai, aj)

    bi = BCM([[0, 1], [0, 5]])
    bj = BCM([[0, 7], [0, 11]])
    print("\nScenario b")
    print_scenario(bi, bj)

    alpha_i = BCM([[1, 1], [0, 5]])
    alpha_j = BCM([[1, 7], [0, 11]])
    print("\nScenario alpha")
    print_scenario(alpha_i, alpha_j)

    # add one false positive to each

    ai[1, 0] += 1
    aj[1, 0] += 1
    print("\nScenario a+1")
    print_scenario(ai, aj)

    bi[1, 0] += 1
    bj[1, 0] += 1
    print("\nScenario b+1")
    print_scenario(bi, bj)

    alpha_i[1, 0] += 1
    alpha_j[1, 0] += 1
    print("\nScenario alpha+1")
    print_scenario(alpha_i, alpha_j)

    # TE has mostly correct indicate bias
    c_i = BCM([[1000, 1], [20, 0]])
    c_j = BCM([[1000, 20], [1, 0]])
    print("\nScenario C")
    print_scenario(c_i, c_j)

    # Just as much bias as this where its very clear that i has more bias here than previously
    d_i = BCM([[0, 1], [20, 0]])
    d_j = BCM([[1000, 20], [1, 0]])
    print("\nScenario D")
    print_scenario(d_i, d_j)

    # Change to mostly FP has little change in TE, flipping FP and FN may show this, but then you need both ways each time.
    # The flip will show the large leap in TE.
    e_i = BCM([[2, 0], [997, 0]])
    e_j = BCM([[7, 1], [1, 0]])
    print("\nScenario E")
    print_scenario(e_i, e_j)

    f_i = BCM([[2, 0], [1, 0]])
    f_j = BCM([[7, 1], [1, 0]])
    print("\nScenario F")
    print_scenario(f_i, f_j)

    pass
