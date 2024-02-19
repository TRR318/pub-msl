from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
import numpy as np
from skpsl import ProbabilisticScoringList

if __name__ == '__main__':
    X, y = fetch_openml(data_id=43979, return_X_y=True, as_frame=False)
    # X, y = fetch_openml(data_id=42900, return_X_y=True, as_frame=False)

    xtrain, xtest, ytrain, ytest = train_test_split(X, y)
    clf = ProbabilisticScoringList({1}, stage_clf_params=dict(calibration_method="isotonic")).fit(xtrain, ytrain)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.summer(np.linspace(0, 1, len(clf) + 1)))
    ax = PrecisionRecallDisplay.from_estimator(clf, xtest, ytest).ax_
    for i in reversed(range(len(clf))):
        PrecisionRecallDisplay.from_estimator(clf[i], xtest, ytest, ax=ax, label=None)

    plt.plot()
    plt.savefig(f"../results/precrec_curve_iso.png")
