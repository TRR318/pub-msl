import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from skpsl import ProbabilisticScoringList

from experiments.util import DataLoader

# fit psl and get a instance of the calibrator
X, y = DataLoader("data").load("thorax")
X, X_test, y, _ = train_test_split(X, y, test_size=1 / 3, random_state=4)
pipeline = make_pipeline(SimpleImputer(missing_values=-1, strategy="most_frequent"),
                         ProbabilisticScoringList({-3, -2, -1, 1, 2, 3}).fit(X, y))

sns.set_theme(font_scale=1.5, rc={"text.usetex": True})
sns.set_style("whitegrid")
plt.rc("font", **{"family": "serif"})
plt.rcParams["figure.figsize"] = (16, 7.5)


def plot_ci(i, ax, hide_label=False, legend=True):
    psl = pipeline[-1]
    stage = psl[i + 1]
    scores = stage._compute_total_scores(
        X_test, psl.features[: i + 1], psl.scores[: i + 1], psl.thresholds[: i + 1]
    )
    sigma, idxs = np.unique(scores, return_index=True)

    ls, ps, us = stage.predict_proba(X_test[idxs], ci=.5).T

    ax.errorbar(
        sigma,
        ps,
        np.abs(np.array([ls, us]) - ps), fmt='o', linewidth=2, capsize=6,
        label="Isotonic Regression with 95\% confidence interval" if not hide_label and legend else None,
    )
    ax.axhline(y=1/11, color='black', linestyle=':', label=None if hide_label or not legend else "Decision boundary for $M=10$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if not hide_label:
        ax.set_ylabel(r"$\hat{q}$")
    ax.set_xlabel(f"$T(\mathbf{{x}})$ at Stage {i + 1}")


fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 4, width_ratios=[4, 6, 7, 10])
fig.suptitle("Coronary Heart Disease")

ax0 = None
for i in range(4):
    ax = fig.add_subplot(gs[0, i], **(dict(sharey=ax0) if ax0 is not None else {}))
    if not ax0:
        ax0 = ax
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
    plot_ci(i, ax, hide_label=i > 0, legend=False)
ax = fig.add_subplot(gs[1, :], sharey=ax0)
plot_ci(6, ax)
h, l = plt.gca().get_legend_handles_labels()
h = [h[1], h[0]]
l = [l[1], l[0]]
fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0), frameon=False)

plt.show()
fig.savefig("fig/uncertainty quantification.pdf", bbox_inches="tight")

plt.rcParams["figure.figsize"] = (18, 5)
for i in range(4, 10):
    fig, ax = plt.subplots()
    plt.tight_layout(w_pad=0)

    plot_ci(i, ax)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0), frameon=False)
    plt.show()
