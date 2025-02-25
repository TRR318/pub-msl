from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy


class DataLoader:
    def __init__(self, dir="../../data"):
        self.dir = Path(dir)

    def load(self, dataset, binary=False):
        file = self.dir / f"{dataset}.csv"
        if (self.dir / f"{dataset}.csv").exists():
            df = pd.read_csv(file)
            X = df.iloc[:, :-1].to_numpy()
            y = LabelEncoder().fit_transform(df.iloc[:, -1].to_numpy()).astype(int)
        else:
            X, y = fetch_openml(data_id=dataset, return_X_y=True, as_frame=False)
            match dataset:
                case 41945:  # liver
                    y = np.array(y == "1", dtype=int)
                case 42900:  # breast
                    y = np.array(y == 2, dtype=int)
                case _:
                    y = LabelEncoder().fit_transform(y)
            pd.DataFrame(np.hstack((X, y.reshape(-1, 1)))).to_csv(file, index=False)

        if binary:
            # find 2 most common classes and filter dataset
            classes, count = np.unique(y, return_counts=True)
            if len(classes) > 2:
                selected_classes = classes[np.argsort(-count)[:2]]

                mask = np.isin(y, selected_classes)
                return X[mask], y[mask]
        return X, y

    def stats(self, dataset):
        X,y = self.load(dataset)
        counts = np.unique(y,return_counts=True)[1]
        counts = counts/counts.sum()
        return {"Entropy": entropy(counts, base=len(counts)),
                "Instances": X.shape[0],
                "Features": X.shape[1],
                "Classes": len(np.unique(y))}

if __name__ == '__main__':
    print(len(DataLoader().load(61, binary=True)[1]))
