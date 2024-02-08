from itertools import product, chain
from multiprocessing import Pool

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from tqdm import tqdm

from experiments.util import ResultHandler
from skpsl import ProbabilisticScoringList

def estimator_factory(param):
    clf, *params = param
    match clf:
        case "logistic_regression":
            return LogisticRegression(**dict(params))
        case "psl":
            return ProbabilisticScoringList(**dict(params))
        case _:
            raise ValueError(f"classifier {clf} not defined")


def worker(key):
    dataset, fold, params = key

    X, y = fetch_openml(data_id=dataset, return_X_y=True, as_frame=False)
    clf = estimator_factory(params)
    results = cross_validate(clf, X, y,
                             cv=ShuffleSplit(1, test_size=.33, random_state=fold),
                             n_jobs=1,
                             scoring=["accuracy", "roc_auc"])
    # TODO if clf has stages, than also crossval each stage
    return key, results


def dict_product(prefix, d):
    if not isinstance(prefix, list | tuple):
        prefix = [prefix]
    return [prefix + list(dict(zip(d, t)).items()) for t in product(*d.values())]


if __name__ == '__main__':
    datasets = [43979, 42900]
    splits = 20

    rh = ResultHandler("../results")

    # create searchspace
    clf_params = chain(
        dict_product(prefix="logistic_regression", d=dict(random_state=[42])),
        dict_product(prefix="qreg", d=dict(random_state=[42]))
    )
    grid = product(datasets, range(splits), clf_params)
    grid = list(filter(rh.is_unprocessed, grid))

    # execute
    with Pool(12) as p:
        [rh.write_results(params) for params in tqdm(p.imap_unordered(worker, grid), total=len(grid))]
