from ast import literal_eval
from pathlib import Path
from typing import Optional

import pandas as pd


class ResultHandler():
    def __init__(self, outdir="../results"):
        self.outdir = Path(outdir)

    def write_results(self, params):
        key, result = params
        multikey = self._to_dict(key, repeat=len(list(result.values())[0]))

        try:
            df = pd.read_csv(self._file(key))
        except FileNotFoundError:
            df = pd.DataFrame()
        pd.concat((df, pd.DataFrame(multikey | result))).to_csv(self._file(key), index=False)

    def is_unprocessed(self, key):
        d = self._to_dict(key)
        fold = d["fold"]
        params = d["params"]
        # check if fold was already processed
        try:
            df = pd.read_csv(self._file(key))
            keys = set(
                (row["fold"], frozenset(literal_eval(row["params"]).items())) for _, row in
                df[["fold", "params"]].iterrows())
            if (fold, frozenset(params.items())) in keys:
                return False
        except FileNotFoundError:
            pass
        return True

    def _file(self, key):
        dataset = self._to_dict(key)["dataset"]
        return self.outdir / f"results_{dataset}.csv"

    def _to_dict(self, key, *, repeat: Optional[int] = None):
        data_name, fold, params = key
        clf, *hyperparam = params
        df = pd.DataFrame([dict(dataset=data_name, fold=fold, params=dict(hyperparam), clf=clf) | dict(hyperparam)])
        if repeat is not None:
            df = pd.concat([df] * repeat)
            return df.to_dict(orient="list")
        return {k: v[0] for k, v in df.to_dict(orient="list").items()}
