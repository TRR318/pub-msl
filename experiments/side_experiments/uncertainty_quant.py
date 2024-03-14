from experiments.util import DataLoader
from skpsl import ProbabilisticScoringList
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import beta

# for each stage create one plot
# for each total score, we estimate a probability and a upper and lower conficence bound
# the bounds are shown in the plot
sns.set_theme(font_scale=1.5, rc={"text.usetex": True})
sns.set_style("whitegrid")
plt.rc("font", **{"family": "serif"})
plt.rcParams["figure.figsize"] = (18, 5)

X, y = DataLoader("data").load("thorax")
score_set = {-3, -2, -1, 1, 2, 3}

psl = ProbabilisticScoringList(score_set).fit(X, y)
calibrator = psl[1].calibrator  # type: IsotonicRegression


fig, axes = plt.subplots(1, 4, sharey=True, width_ratios=[3, 5, 7, 8])
plt.tight_layout(w_pad=0.05)

for i, ax in enumerate(axes):
    scores = psl[i + 1]._compute_total_scores(
        X, psl.features[: i + 1], psl.scores[: i + 1], psl.thresholds[: i + 1]
    )
    probas = calibrator.fit_transform(scores, y)
    sigma = np.unique(scores)

    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    ls, us = [], []
    for i_ in sigma:

        c = {0: 0, 1: 0} | {
            c_: count
            for c_, count in zip(
                *np.unique(y[scores.squeeze() == i_], return_counts=True)
            )
        }
        pos = c[1]
        neg = c[0]

        a = 0.025
        # 95% binomial proportion ci bounds
        l, u = beta.ppf([a, 1 - a], [pos, pos + 1], [neg + 1, neg])
        p = calibrator.transform([i_])
        # make sure the bounds are sensible wrt. proba estimate
        ls.append(min(np.nan_to_num(l, nan =0), p))
        us.append(max(np.nan_to_num(u, nan=1), p))

    # make sure the bounds are monotonic in the scoreset
    ls = [max(a, b) for a, b in zip([0] + ls, ls)]
    us = list(reversed([min(a, b) for a, b in list(zip([1] + us[::-1], us[::-1]))]))

    ax.scatter(
        sigma,
        calibrator.transform(sigma),
        label="Isotonic Regression" if i == 0 else None,
    )
    ax.scatter(sigma, ls, c="b", marker="_")
    ax.scatter(sigma, us, c="b", marker="_")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    if i == 0:
        ax.set_ylabel(r"$\hat{q}$")
    ax.set_xlabel(f"T(x) at Stage {i+1}")
fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0), frameon=False)
plt.show()
fig.savefig("fig/uncertainty quantification.pdf", bbox_inches="tight")
