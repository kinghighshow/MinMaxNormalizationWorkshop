import pandas as pd


class MinMaxNormalizer:
    """Min-max scaling written from the formula, without numpy or sklearn."""

    def __init__(self, columns, feature_range=(0.0, 1.0)):
        self.columns = list(columns)
        self.low, self.high = feature_range
        self.minimums = {}
        self.maximums = {}

    def fit(self, df):
        for col in self.columns:
            values = df[col].tolist()
            self.minimums[col] = min(values)
            self.maximums[col] = max(values)
        return self

    def _scale(self, value, col):
        span = self.maximums[col] - self.minimums[col]
        if span == 0:
            return self.low
        unit = (value - self.minimums[col]) / span
        return self.low + unit * (self.high - self.low)

    def transform(self, df, suffix="_norm", inplace=False):
        out = df.copy()
        for col in self.columns:
            target = col if inplace else f"{col}{suffix}"
            out[target] = [self._scale(v, col) for v in df[col].tolist()]
        return out

    def fit_transform(self, df, suffix="_norm", inplace=False):
        return self.fit(df).transform(df, suffix=suffix, inplace=inplace)

    def inverse(self, value, col):
        unit = (value - self.low) / (self.high - self.low)
        return self.minimums[col] + unit * (self.maximums[col] - self.minimums[col])

    def summary(self):
        rows = []
        for col in self.columns:
            lo, hi = self.minimums[col], self.maximums[col]
            rows.append({"Feature": col, "Min": lo, "Max": hi, "Range": hi - lo})
        return pd.DataFrame(rows).set_index("Feature")

    def worked_example(self, row):
        rows = []
        for col in self.columns:
            raw = row[col]
            rows.append({
                "Feature": col,
                "Raw": raw,
                "Min": self.minimums[col],
                "Max": self.maximums[col],
                "Normalized": round(self._scale(raw, col), 4),
            })
        return pd.DataFrame(rows).set_index("Feature")

    def verify(self, df_scaled, suffix="_norm"):
        report = {}
        for col in self.columns:
            vals = df_scaled[f"{col}{suffix}"].tolist()
            report[col] = {"min": min(vals), "max": max(vals)}
        return pd.DataFrame(report).T
