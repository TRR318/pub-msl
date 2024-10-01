import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from skpsl import ProbabilisticScoringList
from sklearn.metrics import confusion_matrix
import networkx as nx
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from experiments.util import DataLoader

# fit psl and get a instance of the calibrator


def conservative_weighted_loss(y_true, y_prob, m=10, *, sample_weight=None):
    if y_prob.shape[1] == 3:
        lb, _, ub = y_prob.T
    else:
        ub = y_prob[:, 1]
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        1 - ub < m * ub,
        sample_weight=sample_weight,
        normalize="all",
        labels=[False, True],
    ).ravel()
    return fp + m * fn


def create_dag(psl):
    dag = nx.DiGraph()
    for stage, (first, second) in enumerate(zip(psl, psl[1:])):
        tscores = first.class_counts_per_score.keys()
        score = second.scores[-1]
        for tscore in tscores:
            first_neg = first.class_counts_per_score[tscore][0]
            first_pos = first.class_counts_per_score[tscore][1]
            first_proba = first.calibrator.transform(np.array([tscore])).item()
            second_0_neg = second.class_counts_per_score[tscore][0]
            second_0_pos = second.class_counts_per_score[tscore][1]
            second_0_proba = second.calibrator.transform(np.array([tscore])).item()
            second_1_neg = second.class_counts_per_score[tscore + score][0]
            second_1_pos = second.class_counts_per_score[tscore + score][1]
            second_1_proba = second.calibrator.transform(
                np.array([tscore + score])
            ).item()

            dag.add_node(
                (stage, tscore),
                num_neg=first_neg,
                num_pos=first_pos,
                proba=first_proba,
            )
            dag.add_node(
                (stage + 1, tscore),
                num_neg=second_0_neg,
                num_pos=second_0_pos,
                proba=second_0_proba,
            )
            dag.add_node(
                (stage + 1, tscore + score),
                num_neg=second_1_neg,
                num_pos=second_1_pos,
                proba=second_1_proba,
            )
            # case 1: Feature ist not active
            dag.add_edge((stage, tscore), (stage + 1, tscore), score=0)
            # case 1: Feature ist active
            dag.add_edge((stage, tscore), (stage + 1, tscore + score), score=score)
    return dag


def expected_loss(y_prob, m=10):
    ub = y_prob[:, 1].item()
    y_decision = 1 - ub < m * ub
    loss = (1 - ub) * 1 if y_decision else ub * 10
    return loss


def worker(
    seed,
    per_instance_budget=3,
    ci=0.5,
    k=0.12,
    score_set={-3, -2, -1, 1, 2, 3},
):
    X, y = DataLoader("../data").load("thorax")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=seed
    )
    pipeline = make_pipeline(
        SimpleImputer(missing_values=-1, strategy="most_frequent"),
        ProbabilisticScoringList(score_set=score_set).fit(X, y),
    )

    pipeline.fit(X_train, y_train)
    imputer = pipeline[0]
    psl = pipeline[1]

    X_test = imputer.transform(X_test)

    overall_budget = per_instance_budget * len(y_test)
    # Predictions when always evaluation the same amount of features per instance
    y_prob_non_adaptive = pipeline.predict_proba(X_test, k=per_instance_budget, ci=ci)

    # Instantiate dag for heuristic
    dag = create_dag(psl=psl)
    y_prob_adaptive = []
    remaining_budget = overall_budget
    imputer = pipeline[0]
    psl = pipeline[1]

    X_test = imputer.transform(X_test)

    decision_threshold = 1 / 11
    wloss_lower_confidence = decision_threshold - decision_threshold * k
    wloss_upper_confidence = decision_threshold + (1 - decision_threshold) * k

    for x_test in X_test:
        x_test = x_test.reshape(1, -1)
        stage = 0
        current_proba = psl.predict_proba(x_test, k=stage, ci=ci).squeeze()

        while (
            wloss_lower_confidence
            < current_proba[2 if ci else 1]
            < wloss_upper_confidence
            and remaining_budget > 0
            and stage + 1 < len(psl)
        ):
            stage += 1
            current_proba = psl.predict_proba(x_test, k=stage, ci=ci).squeeze()
            remaining_budget -= 1
        y_prob_adaptive.append(current_proba)
    y_prob_adaptive = np.stack(y_prob_adaptive)

    cwloss_adap = conservative_weighted_loss(y_test, y_prob_adaptive)
    cwloss_non_adap = conservative_weighted_loss(y_test, y_prob_non_adaptive)

    if y_prob_non_adaptive.shape[1] == 3:
        lb, _, ub_non_adap = y_prob_non_adaptive.T
    else:
        lb, ub_non_adap = y_prob_non_adaptive.T
    tn_non_adap, fp_non_adap, fn_non_adap, tp_non_adap = confusion_matrix(
        y_test,
        1 - ub_non_adap < 10 * ub_non_adap,
        sample_weight=None,
        normalize=None,
        labels=[False, True],
    ).ravel()

    if y_prob_adaptive.shape[1] == 3:
        lb, _, ub_adap = y_prob_adaptive.T
    else:
        lb, ub_adap = y_prob_adaptive.T

    tn_adap, fp_adap, fn_adap, tp_adap = confusion_matrix(
        y_test,
        1 - ub_adap < 10 * ub_adap,
        sample_weight=None,
        normalize=None,
        labels=[False, True],
    ).ravel()

    # bacc_adap = balanced_accuracy_score(y_test, y_prob_adaptive.argmax(axis=1))
    # bacc_non_adap = balanced_accuracy_score(y_test, y_prob_non_adaptive.argmax(axis=1))

    # acc_adap = accuracy_score(y_test, y_prob_adaptive.argmax(axis=1))
    # acc_non_adap = accuracy_score(y_test, y_prob_non_adaptive.argmax(axis=1))
    return [
        [
            "adaptive",
            seed,
            per_instance_budget,
            overall_budget,
            score_set,
            ci,
            cwloss_adap,
            remaining_budget,
            tp_adap,
            fp_adap,
            fn_adap,
            tn_adap,
        ],
        [
            "baseline",
            seed,
            per_instance_budget,
            overall_budget,
            score_set,
            ci,
            cwloss_non_adap,
            0,
            tp_non_adap,
            fp_non_adap,
            fn_non_adap,
            tn_non_adap,
        ],
    ]


if __name__ == "__main__":
    data = []
    for seed in tqdm(range(0, 100)):
        data += worker(seed)
    df = pd.DataFrame(
        data,
        columns=[
            "method",
            "seed",
            "per_instance_budget",
            "overall_budget",
            "score_set",
            "ci",
            "cwloss",
            "remaining_budget",
            "tp",
            "fp",
            "fn",
            "tn",
        ],
    )
    df.to_csv("../results/cost_results_naive.csv")
