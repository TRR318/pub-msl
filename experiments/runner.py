from itertools import product, chain
from multiprocessing import Pool

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline, Pipeline
from tqdm import tqdm


from util import ResultHandler
from skpsl import ProbabilisticScoringList
from skpsl.preprocessing.binarizer import MinEntropyBinarizer


def estimator_factory(param):
    clf, *params = param
    match clf:
        case "pipeline":
            kwargs = dict(params)
            pre = MinEntropyBinarizer(method=kwargs["method"])
            kwargs.pop("method")
            psl = ProbabilisticScoringList(**kwargs)
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
    results["expected_entropy_train"] = [est.score(X_train, y_train)]
    results["expected_entropy_test"] = [est.score(X_test, y_test)]
    if isinstance(est, Pipeline):
        pre = est.named_steps["minentropybinarizer"]
        psl = est.named_steps["probabilisticscoringlists"]
        X_train = pre.transform(X_train)
        X_test = pre.transform(X_test)
    else:
        psl = est
    keys = [key + (-1,)]
    resultss = [results]
    for k, stage_clf in enumerate(est.stage_clfs):
        cur_results = {}
        cur_results["expected_entropy_train"] = [stage_clf.score(X_train, y_train)]
        cur_results["expected_entropy_test"] = [stage_clf.score(X_test, y_test)]
        cur_results["fit_time"] = [-1]
        cur_results["score_time"] = [-1]
        y_pred = stage_clf.predict(X_test)

        cur_results["test_accuracy"] = [accuracy_score(y_test, y_pred)]
        cur_results["test_roc_auc"] = [
            roc_auc_score(y_test, stage_clf.predict_proba(X_test)[:, 1])
        ]
        keys.append(key + (k,))
        resultss.append(cur_results)
    # TODO if clf has stages, than also crossval each stage
    del results["estimator"]
    del results["indices"]
    return zip(keys, resultss)


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
        # dict_product(
        #     prefix="pipeline", d=dict(score_set=[score_set], method=["bisect", "brute"])
        # ),
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
            for params in chain.from_iterable(
                tqdm(p.imap_unordered(worker, grid), total=len(grid))
            )
        ]
