from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    for dataset in [43979, 42900]:
        metric = "accuracy"  # "roc_auc"
        print(dataset)
        X, y = fetch_openml(data_id=dataset, return_X_y=True, as_frame=False)
        print(cross_val_score(LogisticRegression(max_iter=500), X, y, n_jobs=-1, scoring=metric).mean())