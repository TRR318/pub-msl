from itertools import product

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import networkx as nx
import numpy as np
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed

from skpsl.estimators import ProbabilisticScoringSystem, ProbabilisticScoringList
from experiments.util import DataLoader

scoreset = [0, 1, 2]
dataset = "thorax_filtered"

X, y = DataLoader("data").load(dataset)
wpos = y.mean()


def from_scorevec(scores):
    scores = np.array(scores)
    features = np.nonzero(scores)[0]
    return list(features), list(scores[features])


@delayed
def fit_predict(scores_):
    features, scores = from_scorevec(scores_)

    score = (
        ProbabilisticScoringSystem(features, scores=list(scores)).fit(X, y).score(X, y)
    )

    return scores_, score


G = nx.Graph()
node = dict()

for scores, result in tqdm(
        Parallel(n_jobs=12, return_as="generator")(
            fit_predict(scores_) for scores_ in product(scoreset, repeat=X.shape[1])
        ),
        total=len(scoreset) ** X.shape[1],
):
    id_ = np.count_nonzero(scores), round(result, 3)
    node[tuple(scores)] = id_

    for index in np.nonzero(scores)[0]:
        # get previous nodes by removing one non-zero feature
        from_ = np.array(scores)
        from_[index] = 0

        G.add_edge(node[tuple(from_)], id_)

pos = {p: p for p in G.nodes}

sns.set(font_scale=1.5, rc={"text.usetex": True})
sns.set_style("whitegrid")
plt.rc("font", **{"family": "serif"})
fig, ax = plt.subplots(layout="constrained")
fig.set_size_inches(12, 4)
ax.set_ylabel("Expected Entropy")
ax.set_xlabel("Stage")

print("drawing edges")
edges = nx.draw_networkx_edges(
    G, pos, edge_color="#a7bad8", ax=ax, width=0.5, node_size=0, alpha=.1
)
edges.set_zorder(0)
plt.grid(alpha=.2, c="black")

# highlight cascade
print("fitting cascade")
psl = ProbabilisticScoringList(score_set=set(scoreset) - {0}).fit(X, y)
cascade = [(i, round(clf.score(X, y), 3)) for i, clf in enumerate(psl.stage_clfs)]

G = nx.Graph()
for u, v in zip(cascade, cascade[1:]):
    G.add_edge(u, v)
print("drawing cascade")
nx.draw_networkx_nodes(G, pos, node_color="#1f78b4", node_size=50, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color="#1f78b4", ax=ax, width=1, node_size=10)

# plt.box(False)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_xlim(-0.2, X.shape[1] + 0.2)

print("generating file")
fig.suptitle("Coronary Heart Disease")
# plt.show()

fig.savefig(f"fig/{dataset}_greedy_search.pdf", bbox_inches="tight")
