import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from skqreg.estimators import QuantisticRegression


def load_data(id_):
    X, y = fetch_openml(data_id=id_, return_X_y=True, as_frame=False)
    if set(y) == {"True", "False"}:
        y = np.array(y == "True", dtype=int)
    return X, y


if __name__ == '__main__':
    for dataset in [43979, 42900]:
        metric = "accuracy"#"roc_auc"
        X, y = load_data(dataset)
        print(cross_val_score(LogisticRegression(max_iter=500), X, y, n_jobs=-1, scoring=metric).mean())
        print(cross_val_score(QuantisticRegression(), X, y, n_jobs=-1, scoring=metric).mean())
