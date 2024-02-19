from itertools import product, chain
from multiprocessing import Pool

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.metrics import make_scorer, get_scorer, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm

from experiments.util import DataLoader
from util import ResultHandler
from skpsl import ProbabilisticScoringList
from skpsl.preprocessing.binarizer import MinEntropyBinarizer
from skpsl.metrics import expected_entropy_loss, weighted_loss

RESULTFOLDER = "results"
DATAFOLDER = "data"


def estimator_factory(param):
    clf, *params = param
    params = [(name, dict((v,))) if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) else (name, v) for
              name, v in params]
    match clf:
        case "psl_prebin":
            kwargs = dict(params)
            pre = MinEntropyBinarizer(method=kwargs["method"])
            kwargs.pop("method")
            psl = ProbabilisticScoringList(**kwargs)
            return make_pipeline(pre, psl)
        case "psl":
            psl = make_pipeline(FunctionTransformer(), ProbabilisticScoringList(**dict(params)))
            return psl
        case _:
            raise ValueError(f"classifier {clf} not defined")


def worker(key):
    fold, dataset, params = key
    rh = ResultHandler(RESULTFOLDER)
    rh.register_run(key)

    scoring = dict(acc=get_scorer("accuracy"),
                   bacc=get_scorer("balanced_accuracy"),
                   roc=get_scorer("roc_auc"),
                   brier=get_scorer("neg_brier_score"),
                   ent=make_scorer(lambda _, pred: expected_entropy_loss(pred), response_method="predict_proba"),
                   wloss=make_scorer(weighted_loss, response_method="predict_proba")
                   )

    X, y = DataLoader(DATAFOLDER).load(dataset)

    clf = estimator_factory(params)
    results = cross_validate(
        clf,
        X,
        y,
        cv=ShuffleSplit(1, test_size=0.33, random_state=fold),
        n_jobs=1,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
    )

    (est,) = results["estimator"]
    indices = results["indices"]
    del results["estimator"]
    del results["indices"]

    (train,) = indices["train"]
    (test,) = indices["test"]
    y_train, y_test = y[train], y[test]
    X_train = est[:-1].transform(X[train])
    X_test = est[:-1].transform(X[test])
    psl = est[-1]

    results = pd.DataFrame(results).to_dict("records")
    for k, stage in enumerate(psl):
        cur_results = (
                {f"train_{name}": scorer(stage, X_train, y_train) for name, scorer in scoring.items()} |
                {f"test_{name}": scorer(stage, X_test, y_test) for name, scorer in scoring.items()} |
                dict(stage=k))
        results.append(cur_results)

        if k > 0:
            X_train_ = X_train[:, stage.features]
            X_test_ = X_test[:, stage.features]
            logreg = LogisticRegression(max_iter=1000).fit(X_train_, y_train)
            cur_results = (
                    {f"train_{name}": scorer(logreg, X_train_, y_train) for name, scorer in scoring.items()} |
                    {f"test_{name}": scorer(logreg, X_test_, y_test) for name, scorer in scoring.items()} |
                    dict(stage=k) |
                    dict(clf_variant="logreg"))
            results.append(cur_results)
    rh.write_results(key, results)


def dict_product(prefix, d):
    if not isinstance(prefix, list | tuple):
        prefix = [prefix]
    return [tuple(prefix + list(dict(zip(d, t)).items())) for t in product(*d.values())]


if __name__ == "__main__":
    datasets = ["thorax", 41945, 43979, 42900]
    splits = 20

    rh = ResultHandler(RESULTFOLDER)
    rh.clean()

    full_scores = (-3, -2, -1, 1, 2, 3)
    score_sets = [(1,), (1, 2), (1, 2, 3), full_scores]
    # create searchspace
    clf_params = chain(
        dict_product(
            prefix="psl_prebin", d=dict(score_set=[full_scores], lookahead=[1], method=["bisect", "brute"],
                                        stage_clf_params=[("calibration_method", "isotonic")])
        ),
        dict_product(
            prefix="psl", d=dict(score_set=[full_scores], lookahead=[1], method=["bisect", "brute"],
                                 stage_clf_params=[("calibration_method", "isotonic")])
        ),
        dict_product(
            prefix="psl", d=dict(score_set=[full_scores], lookahead=[1], method=["bisect"],
                                 stage_clf_params=[("calibration_method", "isotonic")
                                     , ("calibration_method", "beta")])
        ),
        # dict_product(
        #    prefix="psl", d=dict(score_set=[full_scores], lookahead=[2], method=["bisect"],
        #                         stage_clf_params=[("calibration_method", "isotonic")])
        # ),
        dict_product(
            prefix="psl_prebin", d=dict(score_set=score_sets, lookahead=[1], method=["bisect"],
                                        stage_clf_params=[("calibration_method", "isotonic")])
        ),
    )
    grid = list(dict.fromkeys(product(range(splits), datasets, clf_params)))
    grid = list(filter(rh.is_unprocessed, grid))

    # execute
    with Pool(12) as p:
        list(tqdm(p.imap_unordered(worker, grid), total=len(grid)))
