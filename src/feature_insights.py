import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class FeatureInsights:
    """Covariance, distance dominance and price impact for raw vs normalized features."""

    def __init__(self, raw, normalized, target="Price", seed=42):
        self.raw = raw
        self.normalized = normalized
        self.target = target
        self.rng = np.random.default_rng(seed)

    def matrix(self, kind="normalized", features=None, min_abs=0.0):
        cols = features or list(self.raw.columns)
        if kind == "raw":
            m = self.raw[cols].cov()
        elif kind == "normalized":
            m = self.normalized[cols].cov()
        elif kind == "correlation":
            m = self.raw[cols].corr()
        else:
            raise ValueError("kind must be 'raw', 'normalized' or 'correlation'")
        if min_abs > 0:
            strength = self.raw[cols].corr().abs()
            keep = (strength >= min_abs) | np.eye(len(cols), dtype=bool)
            m = m.where(keep)
        return m

    def strongest_pairs(self, n=5):
        corr = self.raw.corr()
        pairs = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                pairs.append((a, b, corr.loc[a, b]))
        out = pd.DataFrame(pairs, columns=["Feature A", "Feature B", "r"])
        return out.reindex(out["r"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)

    def distance_share(self, pairs=500):
        idx = self.rng.choice(len(self.raw), size=(pairs, 2))
        shares = {}
        for name, frame in [("Raw", self.raw), ("Min-max", self.normalized)]:
            values = frame.values.astype(float)
            sq = (values[idx[:, 0]] - values[idx[:, 1]]) ** 2
            shares[name] = sq.sum(axis=0) / sq.sum()
        return pd.DataFrame(shares, index=self.raw.columns)

    def price_impact(self, boots=500):
        X = self.normalized.drop(columns=self.target)
        y = self.raw[self.target]
        model = LinearRegression().fit(X, y)
        draws = []
        n = len(X)
        for _ in range(boots):
            pick = self.rng.integers(0, n, n)
            draws.append(LinearRegression().fit(X.iloc[pick], y.iloc[pick]).coef_)
        draws = np.array(draws)
        table = pd.DataFrame({
            "Impact ($, min to max)": model.coef_,
            "CI low": np.percentile(draws, 2.5, axis=0),
            "CI high": np.percentile(draws, 97.5, axis=0),
            "Correlation r": [np.corrcoef(X[c], y)[0, 1] for c in X.columns],
        }, index=X.columns)
        table["CI crosses 0"] = (table["CI low"] < 0) & (table["CI high"] > 0)
        self.r2 = model.score(X, y)
        return table.reindex(table["Impact ($, min to max)"].abs().sort_values(ascending=False).index)
