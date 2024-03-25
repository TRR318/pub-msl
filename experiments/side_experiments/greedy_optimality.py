from itertools import product

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from joblib import Parallel, delayed

from skpsl.estimators import ProbabilisticScoringSystem, ProbabilisticScoringList
from experiments.util import DataLoader

scoreset = [0, 1, 2, 3]
dataset = "thorax"

X, y = DataLoader("data").load(dataset)
X = SimpleImputer(missing_values=-1, strategy="most_frequent").fit_transform(X, y)


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
ax.set_xlabel("Stage / selected Feature in Stage")

print("drawing edges")
edges = nx.draw_networkx_edges(
    G, pos, edge_color="#a7bad8", ax=ax, width=0.5, node_size=0, alpha=0.1
)
edges.set_zorder(0)
plt.grid(alpha=0.2, c="black")

# highlight cascade
print("fitting cascade")
psl = ProbabilisticScoringList(score_set=set(scoreset) - {0}).fit(X, y)
cascade = [(i, round(clf.score(X, y), 3)) for i, clf in enumerate(psl)]

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

ax.set_xticks(list(range(len(psl))))
ax.set_xticklabels(
    [
        f"{i}\n$f_{{{feature+1}}}$" if feature is not None else f"{i}"
        for i, feature in enumerate([None] + psl.features)
    ]
)

print("generating file")
fig.suptitle("Coronary Heart Disease")
# plt.show()

fig.savefig(f"fig/{dataset}_greedy_search.pdf", bbox_inches="tight")

result = {}

# Iterate through the dictionary
for key, (v1, v2) in node.items():
    # If v1 is not in result or the current v2 is greater than the stored v2
    if v1 not in result or v2 < node[result[v1][1]][1]:
        # Convert the key tuple to a list of (index, value) pairs for non-zero entries
        key_as_list = [(index, value) for index, value in enumerate(key) if value != 0]
        result[v1] = (key_as_list, key)  # Store both formats for later use

# Extract and format the final result
final_result = {v1: key_list for v1, (key_list, original_key) in result.items()}

print(final_result)