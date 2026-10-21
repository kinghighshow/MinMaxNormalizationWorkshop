import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAnalyzer:
    """PCA from first principles (covariance -> eigen-decomposition), checked against sklearn."""

    def __init__(self, X, target=None):
        self.X = X.copy()
        self.features = list(X.columns)
        self.target = target
        self.labels = []

    def fit(self):
        centered = self.X - self.X.mean()
        self.covariance = pd.DataFrame(
            np.cov(centered.values, rowvar=False), index=self.features, columns=self.features
        )
        eigvals, eigvecs = np.linalg.eigh(self.covariance.values)
        order = np.argsort(eigvals)[::-1]
        self.eigenvalues = eigvals[order]
        vectors = eigvecs[:, order]

        self.sk = PCA().fit(self.X.values)
        for i in range(vectors.shape[1]):
            if np.dot(vectors[:, i], self.sk.components_[i]) < 0:
                vectors[:, i] *= -1
        self.eigenvectors = vectors

        self.labels = [f"PC{i + 1}" for i in range(len(self.features))]
        self.ratio = self.eigenvalues / self.eigenvalues.sum()
        self.cumulative = np.cumsum(self.ratio)
        self.scores = pd.DataFrame(centered.values @ vectors, columns=self.labels, index=self.X.index)
        return self

    def agreement(self):
        return pd.DataFrame({
            "Manual eigen ratio": self.ratio.round(4),
            "sklearn ratio": self.sk.explained_variance_ratio_.round(4),
            "Abs. difference": np.abs(self.ratio - self.sk.explained_variance_ratio_),
        }, index=self.labels)

    def variance_table(self):
        return pd.DataFrame({
            "Eigenvalue": self.eigenvalues.round(5),
            "Explained %": (self.ratio * 100).round(2),
            "Cumulative %": (self.cumulative * 100).round(2),
        }, index=self.labels)

    def components_for(self, threshold=0.90):
        return int(np.argmax(self.cumulative >= threshold) + 1)

    def loadings(self):
        return pd.DataFrame(self.eigenvectors, index=self.features, columns=self.labels)

    def top_contributors(self, pc, n=2):
        col = self.loadings()[pc]
        top = col.abs().sort_values(ascending=False).head(n).index
        return [(f, round(float(col[f]), 3)) for f in top]

    def target_correlation(self):
        if self.target is None:
            raise ValueError("No target supplied")
        corr = [np.corrcoef(self.scores[pc], self.target)[0, 1] for pc in self.labels]
        return pd.Series(corr, index=self.labels, name="Correlation with Price")

    def most_target_related(self):
        corr = self.target_correlation()
        return corr.abs().idxmax(), corr[corr.abs().idxmax()]

    def variance_importance(self):
        weights = (self.loadings() ** 2) * self.ratio
        share = weights.sum(axis=1)
        return (share / share.sum()).sort_values(ascending=False).rename("Variance share")

    @staticmethod
    def compare_scalings(raw_X, minmax_X):
        options = {
            "Raw units": raw_X.values,
            "Min-max [0, 1]": minmax_X.values,
            "Standardized (z-score)": StandardScaler().fit_transform(raw_X.values),
        }
        rows = {}
        for name, values in options.items():
            cumulative = np.cumsum(PCA().fit(values).explained_variance_ratio_)
            rows[name] = cumulative
        table = pd.DataFrame(rows, index=[f"PC{i + 1}" for i in range(raw_X.shape[1])])
        return table
