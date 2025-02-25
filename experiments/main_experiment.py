import logging
from itertools import product, chain

import numpy as np
import pandas as pd
import xgboost as xgb
from calibration import get_ece as expected_calibration_loss
from joblib import delayed, wrap_non_picklable_objects, Parallel
from mapie.metrics import classification_coverage_score
from miss import MISSClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    get_scorer,
    confusion_matrix,
    recall_score, balanced_accuracy_score,
)
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.pipeline import make_pipeline
from skpsl import ProbabilisticScoringList, MulticlassScoringList
from skpsl.metrics import expected_entropy_loss, ambiguity_aware_accuracy
from skpsl.preprocessing.binarizer import MinEntropyBinarizer
from tqdm import tqdm

from experiments.util.nb_wrapper import StagedNBClassifier
from experiments.util.timer import Timer
from util import DataLoader
from util import ResultHandler

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

RESULTFOLDER, miss_timeout = "results_90s_miss", 90
#RESULTFOLDER, miss_timeout = "results", 30 * 60
DATAFOLDER = "data"


def estimator_factory(param):
    clf, *params = param
    params = [
        (
            (name, dict((v,)))
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str)
            else (name, v)
        )
        for name, v in params
    ]
    match clf:
        case "miss":
            kwargs = dict(params)
            pre = MinEntropyBinarizer()
            psl = MISSClassifier(**kwargs)
            return make_pipeline(
                SimpleImputer(missing_values=-1, strategy="most_frequent"), pre, psl
            )

        case "psl_prebin":
            kwargs = dict(params)
            pre = MinEntropyBinarizer()
            psl = ProbabilisticScoringList(**kwargs)
            return make_pipeline(
                SimpleImputer(missing_values=-1, strategy="most_frequent"), pre, psl
            )
        case "msl_prebin":
            kwargs = dict(params)
            pre = MinEntropyBinarizer()
            msl = MulticlassScoringList(**kwargs)
            return make_pipeline(
                SimpleImputer(missing_values=-1, strategy="most_frequent"), pre, msl
            )
        case _:
            raise ValueError(f"classifier {clf} not defined")


def conservative_weighted_loss(y_true, y_prob, m=10, *, sample_weight=None):
    lb, _, ub = y_prob.T
    tn, fp, fn, tp = confusion_matrix(
        y_true, 1 - ub < m * ub, sample_weight=sample_weight, normalize="all"
    ).ravel()
    return fp + m * fn


def score(scoring, clf, X_train, y_train, X_test, y_test, additional_params):
    cur_results = dict()
    for name, scorer in scoring.items():
        try:
            cur_results |= {f"train_{name}": scorer(clf, X_train, y_train)}
        except (ValueError, UserWarning, IndexError):
            pass
        try:
            cur_results |= {f"test_{name}": scorer(clf, X_test, y_test)}
        except (ValueError, UserWarning, IndexError):
            pass
    return cur_results | additional_params


def worker_facory():
    def cov(true, pred):
        if len(pred.shape) == 1:
            pred = np.vstack((1 - pred, pred)).T
        covered = pred == pred.max(axis=1, keepdims=True)
        return classification_coverage_score(true, covered)

    def eff(true, pred):
        if len(pred.shape) == 1:
            pred = np.vstack((1 - pred, pred)).T
        covered = pred == pred.max(axis=1, keepdims=True)
        return np.count_nonzero(covered, axis=1).mean()

    rh = ResultHandler(RESULTFOLDER)

    # all scorers that can be run on training and test independendly
    scoring = dict(
        acc=get_scorer("accuracy"),
        aaacc=make_scorer(ambiguity_aware_accuracy, response_method="predict_proba"),
        bacc=make_scorer(lambda true, pred: balanced_accuracy_score(true, pred, adjusted=True),
                         response_method="predict"),
        ece=make_scorer(lambda true, pred: expected_calibration_loss(pred, true), response_method="predict_proba"),
        roc=get_scorer("roc_auc"),
        brier=get_scorer("neg_brier_score"),
        recall=get_scorer("recall"),
        recall_at_wloss=make_scorer(
            lambda true, pred: recall_score(true, 1 - pred < 10 * pred),
            response_method="predict_proba",
        ),
        prec=get_scorer("precision"),
        spec_at_wloss=make_scorer(
            lambda true, pred: recall_score(1 - true, 1 - pred >= 10 * pred),
            response_method="predict_proba",
        ),
        f1=get_scorer("f1"),
        ent=make_scorer(
            lambda _, pred: expected_entropy_loss(pred), response_method="predict_proba"
        ),
        cov=make_scorer(cov, response_method="predict_proba"),
        eff=make_scorer(eff, response_method="predict_proba")
    )

    @delayed
    @wrap_non_picklable_objects
    def worker(fold, dataset, params):
        key = (fold, dataset, params)
        rh.register_run(key)

        clf_name, *_ = params
        X, y = DataLoader(DATAFOLDER).load(dataset)
        pipeline = estimator_factory(params)

        if len(np.unique(y)) > 2 and clf_name == "psl_prebin":
            return

        results = cross_validate(
            pipeline,
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
        results["n_features"] = len(est)

        (train,) = indices["train"]
        (test,) = indices["test"]
        y_train, y_test = y[train], y[test]
        X_train = est[:-1].transform(X[train])
        X_test = est[:-1].transform(X[test])
        clf = est[-1]

        results = pd.DataFrame(results).to_dict("records")

        """
          fit psl -> predict

          fit msl -> predict
          -> feature ordering
          fit NB -> predict with feature seq prefix
          -> for each feature seq prefix
          -> for each remaining clf (lr, xgboost, rf, miss)
              fit classifiers with featuers, predict
        """
        # handle stage classifiers
        match clf_name:
            case "psl" | "psl_prebin":
                for k, stage in enumerate(clf):
                    results.append(score(scoring, stage, X_train, y_train, X_test, y_test, dict(stage=k)))
            case "msl" | "msl_prebin":
                with Timer() as timer:
                    clf_nb = StagedNBClassifier().fit(X_train, y_train)
                elapsed = timer.interval

                for k, stage in enumerate(clf):
                    results.append(score(scoring, stage, X_train, y_train, X_test, y_test, dict(stage=k)))

                    clf = clf_nb[stage.features]
                    if k == 0:
                        results.append(
                            score(scoring, clf, X_train, y_train, X_test, y_test,
                                  dict(stage=k, clf_variant="nb", fit_time=elapsed)))
                    else:
                        results.append(
                            score(scoring, clf, X_train, y_train, X_test, y_test, dict(stage=k, clf_variant="nb")))

                    # compute stage-wise classifiers
                    for name in ["logreg", "xgboost", "random_forest", "miss"]:
                        elapsed = None
                        if k == 0:
                            # add performance of stage clf in to mimic performance of logreg at stage 0 (majority classifier)
                            X_train_ = X_train
                            X_test_ = X_test
                            clf = DummyClassifier().fit(X_train_, y_train)
                        else:
                            X_train_ = X_train[:, stage.features]
                            X_test_ = X_test[:, stage.features]

                            with Timer() as timer:
                                match name:
                                    case "xgboost":
                                        clf = make_pipeline(
                                            SimpleImputer(
                                                missing_values=-1, strategy="most_frequent"
                                            ),
                                            xgb.XGBClassifier(n_jobs=1)
                                        ).fit(X_train_, y_train)
                                    case "random_forest":
                                        clf = make_pipeline(
                                            SimpleImputer(missing_values=-1, strategy="most_frequent"),
                                            RandomForestClassifier(n_jobs=1)
                                        ).fit(X_train_, y_train)
                                    case "miss":
                                        n = len(stage.features)
                                        clf = make_pipeline(
                                            SimpleImputer(missing_values=-1, strategy="most_frequent"),
                                            MISSClassifier(mc_l0_min=n, l0_min=n, l0_max=n, mc_l0_max=n,
                                                           max_intercept=3,
                                                           max_coefficient=3, max_runtime=miss_timeout)
                                        ).fit(X_train_, y_train)
                                    case _:
                                        clf = make_pipeline(
                                            SimpleImputer(missing_values=-1, strategy="most_frequent"),
                                            LogisticRegression(
                                                max_iter=10000, penalty="l2", n_jobs=1
                                            )
                                        ).fit(X_train_, y_train)

                            elapsed = timer.interval

                        results.append(
                            score(scoring, clf, X_train_, y_train, X_test_, y_test,
                                  dict(stage=k, clf_variant=name, fit_time=elapsed)))

        rh.write_results(key, results)

    return worker


def dict_product(prefix, d):
    if not isinstance(prefix, list | tuple):
        prefix = [prefix]
    return [tuple(prefix + list(dict(zip(d, t)).items())) for t in product(*d.values())]


if __name__ == "__main__":
    # diabetes, ilp, bcc
    datasets_bin = [37, 41945, 42900]
    #  61 (iris), fußball, wine
    datasets_mc = [61, "player", 187, "segmentation"]
    splits = 20

    rh = ResultHandler(RESULTFOLDER)
    rh.clean()

    base = dict(
        score_set=[(-3, -2, -1, 0, 1, 2, 3)],
    )

    mc_clfs = list(chain(dict_product(prefix="msl_prebin", d=base),
                         ))
    # original parameter space
    clf_params = list(chain(
        dict_product(prefix="psl_prebin", d=base),
        mc_clfs
    ))

    grid = list(
        filter(
            rh.is_unprocessed,
            dict.fromkeys((split, dataset, param)
                          for split in range(splits)
                          for dataset, param in
                          list(product(datasets_bin, clf_params)) +
                          list(product(datasets_mc, mc_clfs))),
        )
    )

    worker = worker_facory()
    list(
        tqdm(
            Parallel(n_jobs=12, return_as="generator_unordered")(
                worker(fold, dataset, params) for fold, dataset, params in grid
            ),
            total=len(grid),
        )
    )
