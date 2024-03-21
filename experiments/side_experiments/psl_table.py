from skpsl import ProbabilisticScoringList
from experiments.util import DataLoader

X, y = DataLoader("data").load("thorax_filtered")
psl = ProbabilisticScoringList({-3, -2, -1, 1, 2,3}).fit(X, y)

df = psl.inspect(4).iloc[:, 1:]
df.rename(columns={"Feature Index": "Feature"})
print(df.to_latex(index=False, na_rep="-", float_format=lambda x: f"{x:.1f}" if abs(x) < 1 else f"{x:.0f}"))