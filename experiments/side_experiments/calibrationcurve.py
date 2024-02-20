from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import fetch_openml
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
import numpy as np
from skpsl import ProbabilisticScoringList

if __name__ == '__main__':
    X, y = fetch_openml(data_id=43979, return_X_y=True, as_frame=False)
    #X, y = fetch_openml(data_id=42900, return_X_y=True, as_frame=False)

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
    clf = ProbabilisticScoringList({1, 3}, stage_clf_params=dict(calibration_method="beta")).fit(xtrain, ytrain)
    CalibrationDisplay.from_estimator(clf, xtest, ytest).plot()
    plt.savefig(f"calibration_curve.png")

    # 43979, scoreset {1,3}, out of sample, sigmoid -> BAUCHI