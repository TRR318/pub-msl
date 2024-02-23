from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


class DataLoader:
    def __init__(self, dir="../data"):
        self.dir = Path(dir)

    def load(self, dataset):
        file = self.dir / f"{dataset}.csv"
        if (self.dir / f"{dataset}.csv").exists():
            df = pd.read_csv(file)
            return df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()

        X, y = fetch_openml(data_id=dataset, return_X_y=True, as_frame=False)
        match dataset:
            case 41945: # liver
                y = np.array(y == "1", dtype=int)
            case 42900: # breast
                y = np.array(y == 2, dtype=int)
        pd.DataFrame(np.hstack((X, y.reshape(-1, 1)))).to_csv(file, index=False)
        return X, y
