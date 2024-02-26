from hashlib import md5
from pathlib import Path
from typing import Optional

import pandas as pd


class ResultHandler:
    def __init__(self, dir="../results"):
        self.dir = Path(dir)

    def write_results(self, key, result):
        multikey = self._to_dict(key, repeat=len(result))

        filename = self._file(key)
        pd.DataFrame(multikey | pd.DataFrame(result).to_dict("list")).to_csv(filename, index=False)

    def register_run(self, key):
        filename = self._file(key)
        filename.touch()

    def is_unprocessed(self, key):
        filename = self._file(key)
        assert not filename.is_dir()
        return not filename.is_file()

    def clean(self):
        for f in self.dir.iterdir():
            if f.stat().st_size == 0:
                f.unlink()

    def _file(self, key):
        d = self._to_dict(key)
        hash = md5(str(d).encode()).hexdigest()
        del d["params"]
        name = ",".join([f"{k}={v}" for k, v in d.items()]).replace(" ", "")
        name = f"{name[0:190]}—{hash[:16]}"

        return self.dir / f"{name}.csv"

    def _to_dict(self, key, *, repeat: Optional[int] = None):
        fold, data_name, params = key
        clf, *hyperparam = params
        hyperparam = {k: f"{v.__name__ if hasattr(v, '__name__') else v}" for k, v in hyperparam}
        df = pd.DataFrame(
            [
                dict(
                    dataset=data_name,
                    fold=fold,
                    params=hyperparam,
                    clf=clf,
                )
                | hyperparam
            ]
        )
        if repeat is not None:
            df = pd.concat([df] * repeat)
            return df.to_dict(orient="list")
        return {k: v[0] for k, v in df.to_dict(orient="list").items()}
