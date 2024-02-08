from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import RocCurveDisplay

from skpsl import ProbabilisticScoringCurve

if __name__ == '__main__':
    X, y = fetch_openml(data_id=43979, return_X_y=True, as_frame=False)
    #X, y = fetch_openml(data_id=42900, return_X_y=True, as_frame=False)
    X = X[:, 0].reshape(-1, 1)
    clf = QuantizationAgg().fit(X, y)
    roc_display = RocCurveDisplay.from_estimator(clf, X, y).plot()
    plt.savefig("roc_curve.png")
