from experiments.util import DataLoader
from skpsl import ProbabilisticScoringList
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.isotonic import IsotonicRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import beta

# fit psl and get a instance of the calibrator
X, y = DataLoader("data").load("thorax")
psl = ProbabilisticScoringList({-3, -2, -1, 1, 2, 3}).fit(X, y)
calibrator = psl[1].calibrator  # type: IsotonicRegression


sns.set_theme(font_scale=1.5, rc={"text.usetex": True})
sns.set_style("whitegrid")
plt.rc("font", **{"family": "serif"})
plt.rcParams["figure.figsize"] = (18, 5)

def plot_ci(i,ax, add_label=False):
    scores = psl[i + 1]._compute_total_scores(
        X, psl.features[: i + 1], psl.scores[: i + 1], psl.thresholds[: i + 1]
    )
    calibrator.fit(scores, y)
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
        neg, pos = c[0], c[1]

        # 95% binomial proportion ci bounds, scaled by bonferoni correction
        a = 0.05/len(sigma)
        l, u = beta.ppf([a/2, 1 - a/2], [pos, pos + 1], [neg + 1, neg])
        p = calibrator.transform([i_])
        # make sure the bounds are sensible wrt. proba estimate
        ls.append(min(np.nan_to_num(l, nan=0), p))
        us.append(max(np.nan_to_num(u, nan=1), p))

    # make sure the bounds are monotonic in the scoreset
    ls = [max([l] + ls[:i]) for i, l in enumerate(ls)]
    us = [min([u] + us[i:]) for i, u in enumerate(us)]

    ps = calibrator.transform(sigma)
    ax.errorbar(
        sigma,
        ps,
        np.abs(np.array([ls,us])-ps),  fmt='o', linewidth=2, capsize=6,
        label="Isotonic Regression with 95\% confidence interval" if i == 0 or add_label else None,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if i == 0:
        ax.set_ylabel(r"$\hat{q}$")
    ax.set_xlabel(f"$T(\mathbf{{x}})$ at Stage {i+1}")


fig, axes = plt.subplots(1, 4, sharey=True, width_ratios=[3, 5, 7, 8])
plt.tight_layout(w_pad=0)

for i, ax in enumerate(axes):
    plot_ci(i,ax)    
fig.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 0), frameon=False)
plt.show()
fig.savefig("fig/uncertainty quantification.pdf", bbox_inches="tight")

for i in range(4,10):
    fig, ax = plt.subplots()
    plt.tight_layout(w_pad=0)

    plot_ci(i,ax, add_label=True)    
    fig.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 0), frameon=False)
    plt.show()
    if i == 6:
        fig.savefig("fig/uncertainty quantification_stage6.pdf", bbox_inches="tight")
