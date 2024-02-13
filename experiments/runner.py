from itertools import product, chain
from multiprocessing import Pool

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from tqdm import tqdm


from util import ResultHandler
from skpsl import ProbabilisticScoringList
from skpsl.preprocessing.binarizer import MinEntropyBinarizer


def estimator_factory(param):
    clf, *params = param
    match clf:
        case "psl_pre_brute":
            pre = MinEntropyBinarizer(method="brute")
            psl = ProbabilisticScoringList(**dict(params))
            return make_pipeline(pre, psl)
        case "psl_pre_bisect":
            pre = MinEntropyBinarizer(method="bisect")
            psl = ProbabilisticScoringList(**dict(params))
            return make_pipeline(pre, psl)
        case "psl":
            psl = ProbabilisticScoringList(**dict(params))
            return psl
        case _:
            raise ValueError(f"classifier {clf} not defined")


def worker(key):
    dataset, fold, params = key

    X, y = fetch_openml(data_id=dataset, return_X_y=True, as_frame=False)
    clf = estimator_factory(params)
    cv = (ShuffleSplit(1, test_size=0.33, random_state=fold),)
    results = cross_validate(
        clf,
        X,
        y,
        cv=ShuffleSplit(1, test_size=0.33, random_state=fold),
        n_jobs=1,
        scoring=["accuracy", "roc_auc"],
        return_estimator=True,
        return_indices=True,
    )
    (est,) = results["estimator"]
    indices = results["indices"]
    (train,) = indices["train"]
    (test,) = indices["test"]
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    for stage_clf in est.stage_clfs:
        expected_entropy_train = stage_clf.score(X_train)
        expected_entropy_test = stage_clf.score(X_test)

    keys = []
    resultss = []
    # TODO if clf has stages, than also crossval each stage
    del results["estimator"]
    del results["indices"]
    return key, results


def dict_product(prefix, d):
    if not isinstance(prefix, list | tuple):
        prefix = [prefix]
    return [prefix + list(dict(zip(d, t)).items()) for t in product(*d.values())]


if __name__ == "__main__":
    datasets = [42900]
    splits = 3

    rh = ResultHandler("./results")
    score_set = (-1, -2, -3, 1, 2, 3)
    # create searchspace
    clf_params = chain(
        dict_product(prefix="psl_pre_brute", d=dict(score_set=[score_set])),
        dict_product(prefix="psl_pre_bisect", d=dict(score_set=[score_set])),
        dict_product(
            prefix="psl", d=dict(score_set=[score_set], method=["bisect", "brute"])
        ),
    )
    grid = product(datasets, range(splits), clf_params)
    # grid = list(filter(rh.is_unprocessed, grid))
    grid = list(grid)

    # execute
    with Pool(12) as p:
        [
            rh.write_results(params)
            for params in tqdm(p.imap_unordered(worker, grid), total=len(grid))
        ]
