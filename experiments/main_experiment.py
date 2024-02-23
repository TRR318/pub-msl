from itertools import product, chain

import pandas as pd
from joblib import delayed, wrap_non_picklable_objects, Parallel
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


def worker_facory():
    rh = ResultHandler(RESULTFOLDER)
    scoring = dict(acc=get_scorer("accuracy"),
                   bacc=get_scorer("balanced_accuracy"),
                   roc=get_scorer("roc_auc"),
                   brier=get_scorer("neg_brier_score"),
                   ent=make_scorer(lambda _, pred: expected_entropy_loss(pred), response_method="predict_proba"),
                   wloss=make_scorer(weighted_loss, response_method="predict_proba")
                   )

    @delayed
    @wrap_non_picklable_objects
    def worker(fold, dataset, params):
        key = (fold, dataset, params)
        rh.register_run(key)

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
                logreg = LogisticRegression(max_iter=10000).fit(X_train_, y_train)
                cur_results = (
                        {f"train_{name}": scorer(logreg, X_train_, y_train) for name, scorer in scoring.items()} |
                        {f"test_{name}": scorer(logreg, X_test_, y_test) for name, scorer in scoring.items()} |
                        dict(stage=k) |
                        dict(clf_variant="logreg"))
                results.append(cur_results)
        rh.write_results(key, results)

    return worker


def dict_product(prefix, d):
    if not isinstance(prefix, list | tuple):
        prefix = [prefix]
    return [tuple(prefix + list(dict(zip(d, t)).items())) for t in product(*d.values())]


if __name__ == "__main__":
    datasets = ["thorax", 41945, 42900]
    splits = 50

    rh = ResultHandler(RESULTFOLDER)
    rh.clean()

    base = dict(score_set=[(-3, -2, -1, 1, 2, 3)], lookahead=[1], method=["bisect"],
                stage_clf_params=[("calibration_method", "isotonic")])

    # create searchspace
    clf_params = chain(
        dict_product(
            prefix="psl_prebin", d=base | dict(method=["bisect", "brute"])
        ),
        dict_product(
            prefix="psl", d=base | dict(method=["bisect", "brute"])
        ),
        dict_product(
            prefix="psl",
            d=base | dict(stage_clf_params=[("calibration_method", "isotonic"), ("calibration_method", "beta")])
        ),
        #dict_product(
        #   prefix="psl", d=base | dict(lookahead=[1, 2])
        #),
        dict_product(
            prefix="psl_prebin", d=base | dict(score_set=[(-3,-2,-1), (-2,-1), (1,), (1, 2), (1, 2, 3), (-3, -2, -1, 1, 2, 3)])
        ),
    )

    grid = list(filter(rh.is_unprocessed,
                       dict.fromkeys(
                           product(range(splits), datasets, clf_params))))

    worker = worker_facory()
    list(tqdm(
        Parallel(n_jobs=12, return_as="generator_unordered")(
            worker(fold, dataset, params)
            for fold, dataset, params
            in grid
        ), total=len(grid)))
