from skpsl import ProbabilisticScoringList

from experiments.util import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns

RESULTFOLDER = "results"
DATAFOLDER = "data"

sns.set(font_scale=1.5, rc={"text.usetex": True})
sns.set_style("whitegrid")
plt.rc("font", **{"family": "serif"})
plt.rcParams["figure.figsize"] = (18, 5)

X, y = DataLoader(DATAFOLDER).load("thorax")
score_set = {-3, -2, -1, 1, 2, 3}

calibrators = dict()
for variant in ["isotonic", "beta"]:
    psl = ProbabilisticScoringList(
        score_set, stage_clf_params=dict(calibration_method=variant)
    )
    psl.fit(X, y)
    calibrators[variant] = psl[-1].calibrator


scores = psl[-1]._compute_total_scores(X, psl.features, psl.scores, psl.thresholds)
probas = calibrators["isotonic"].fit_transform(scores, y)

fig, ax = plt.subplots()

a, c = np.unique([scores.squeeze(), y], axis=1, return_counts=True)
per_line = 13
lines = c.max() // per_line + 1
offset = np.array(
    [
        np.tile(np.linspace(-1,1,per_line), lines),
        np.repeat(np.linspace(0,1, lines), per_line)
    ]
)
sorting = np.argsort([1, 1.1] @ np.abs(offset))
offset = offset[:, sorting]
offset *= [[0.42], [0.08]]
offset += [[0],[0.01]]
X_shift = np.array(
    [
        a[:, j] - (offset[:, i] * (-1) ** a[1, j])
        for j in range(c.size)
        for i in range(c[j])
    ]
)

ax.scatter(*X_shift.T, s=5, c="gray", label="Datapoints")
# d = np.array([scores.squeeze(),y*1.04-.02], dtype=float)
# d+= np.array([np.random.uniform(-.35,.35, d.shape[1]), np.random.uniform(-.02,.02, d.shape[1])])
# ax.scatter(*d, s=5)
# ax.hist2d(scores.squeeze(), y, bins=(50,20), cmin=0, cmap="Grays")
ax.step(
    calibrators["isotonic"].X_thresholds_,
    calibrators["isotonic"].y_thresholds_,
    label="Isotonic Regression",
)
X_ = np.linspace(scores.min(), scores.max(), 1000)
ax.plot(X_, calibrators["beta"].transform(X_.reshape(-1, 1)), label="Beta Calibration")
ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=11))
fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0), frameon=False)
ax.set_xlabel("$T(x)$")
ax.set_ylabel("$\hat{q}$")
plt.show()
fig.savefig("fig/calibration.pdf", bbox_inches="tight")
