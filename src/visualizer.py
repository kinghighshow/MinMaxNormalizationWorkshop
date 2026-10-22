from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

NAVY = "#1B2A41"
TEAL = "#0F8B8D"
CORAL = "#EC6B56"
AMBER = "#F2A541"
SLATE = "#8A97A8"
PALETTE = [TEAL, CORAL, AMBER, "#4C6EF5", "#9B5DE5", NAVY]
DIVERGING = [[0, "#B83B5E"], [0.5, "#F7F7F7"], [1, "#0F8B8D"]]


class HousingVisualizer:
    """Interactive Plotly charts for the normalization and PCA workshop."""

    def __init__(self, image_dir="images"):
        self.image_dir = Path(image_dir)
        template = go.layout.Template(pio.templates["plotly_white"])
        template.layout.font = dict(family="Inter, Segoe UI, Arial", size=13, color=NAVY)
        template.layout.title = dict(font=dict(size=20, color=NAVY), x=0.02, xanchor="left")
        template.layout.colorway = PALETTE
        template.layout.hoverlabel = dict(bgcolor="white", font_size=12)
        pio.templates["housing"] = template
        pio.templates.default = "housing"

    def _finish(self, fig, title, subtitle=None, height=480):
        text = f"<b>{title}</b>"
        if subtitle:
            text += f"<br><span style='font-size:13px;color:{SLATE}'>{subtitle}</span>"
        fig.update_layout(title=dict(text=text), height=height, margin=dict(t=95, l=70, r=40, b=60))
        return fig

    def export(self, fig, name, width=1100, height=None):
        self.image_dir.mkdir(exist_ok=True)
        try:
            fig.write_image(self.image_dir / f"{name}.png", width=width, height=height or fig.layout.height, scale=2)
            return f"saved {name}.png"
        except Exception as err:
            return f"static export skipped ({type(err).__name__}); install kaleido to save PNGs"

    def scale_gap(self, summary_raw, summary_norm):
        fig = go.Figure()
        fig.add_bar(y=summary_raw.index, x=summary_raw["Range"], orientation="h", name="Raw range",
                    marker_color=CORAL, text=[f"{v:,.0f}" for v in summary_raw["Range"]], textposition="outside")
        fig.add_bar(y=summary_norm.index, x=summary_norm["Range"], orientation="h", name="After min-max",
                    marker_color=TEAL, text=[f"{v:.0f}" for v in summary_norm["Range"]], textposition="outside")
        fig.update_xaxes(type="log", title="Range of values (log scale)")
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
        return self._finish(fig, "Raw features live on wildly different scales",
                            "Price spans ~978,000 units while bathrooms span 2. Min-max puts every feature on a range of 1.")

    def distributions(self, raw, norm):
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                            subplot_titles=("Raw values (log scale)", "Min-max normalized"))
        for i, col in enumerate(raw.columns):
            color = PALETTE[i % len(PALETTE)]
            fig.add_box(y=raw[col], name=col, marker_color=color, boxmean=True, showlegend=False, row=1, col=1)
            fig.add_box(y=norm[col], name=col, marker_color=color, boxmean=True, showlegend=False, row=1, col=2)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_yaxes(range=[-0.05, 1.05], row=1, col=2)
        return self._finish(fig, "Same data, comparable shapes",
                            "Normalization keeps each distribution's shape but moves all of them into [0, 1]", height=520)

    def distance_share(self, shares):
        fig = go.Figure()
        for i, feat in enumerate(shares.index):
            fig.add_bar(y=shares.columns, x=shares.loc[feat] * 100, name=feat, orientation="h",
                        marker_color=PALETTE[i % len(PALETTE)],
                        hovertemplate=f"{feat}: %{{x:.2f}}% of squared distance<extra></extra>")
        fig.update_layout(barmode="stack", xaxis=dict(title="Share of squared Euclidean distance (%)", range=[0, 100]),
                          legend=dict(orientation="h", y=-0.25))
        return self._finish(fig, "Who decides who your nearest neighbour is?",
                            "Averaged over 500 random pairs of houses: raw Price owns almost the whole KNN distance", height=420)

    def covariance_heatmap(self, insights, features=None, min_abs=0.0, start="normalized"):
        kinds = [("raw", "Covariance - raw units"), ("normalized", "Covariance - min-max"), ("correlation", "Correlation (r)")]
        fig = go.Figure()
        for kind, label in kinds:
            m = insights.matrix(kind, features=features, min_abs=min_abs)
            if kind == "correlation":
                zmax = 1
                text = m.map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
            else:
                zmax = np.nanmax(np.abs(m.values))
                text = m.map(lambda v: "" if pd.isna(v) else f"{v:.3g}")
            fig.add_heatmap(z=m.values, x=m.columns, y=m.index, text=text.values, texttemplate="%{text}",
                            colorscale=DIVERGING, zmin=-zmax, zmax=zmax, visible=(kind == start), name=label,
                            hovertemplate="%{y} vs %{x}<br>value: %{z:.4g}<extra></extra>",
                            colorbar=dict(title=dict(text=label.split(" - ")[-1], side="right")))
        buttons = []
        for i, (kind, label) in enumerate(kinds):
            buttons.append(dict(label=label, method="update",
                                args=[{"visible": [j == i for j in range(len(kinds))]}]))
        fig.update_layout(
            updatemenus=[dict(buttons=buttons, active=[k for k, _ in kinds].index(start), x=1, xanchor="right",
                              y=1.13, yanchor="bottom", bgcolor="white", bordercolor=SLATE)],
            yaxis=dict(autorange="reversed"), xaxis=dict(side="bottom"),
        )
        note = f"Cells with |r| below {min_abs} hidden" if min_abs > 0 else "Use the menu to switch between raw, normalized and correlation"
        return self._finish(fig, "How the features move together", note, height=660)

    def covariance_explorer(self, insights):
        import ipywidgets as widgets
        from IPython.display import display

        cols = list(insights.raw.columns)
        kind = widgets.ToggleButtons(options=[("Raw covariance", "raw"), ("Min-max covariance", "normalized"),
                                              ("Correlation", "correlation")], value="normalized")
        picks = widgets.SelectMultiple(options=cols, value=tuple(cols), rows=len(cols), description="Features")
        strength = widgets.FloatSlider(value=0.0, min=0.0, max=0.05, step=0.005, readout_format=".3f",
                                       description="Min |r|", continuous_update=False)
        out = widgets.Output()

        def redraw(*_):
            out.clear_output(wait=True)
            chosen = list(picks.value)
            if len(chosen) < 2:
                with out:
                    print("Pick at least two features")
                return
            fig = self.covariance_heatmap(insights, features=chosen, min_abs=strength.value, start=kind.value)
            fig.update_layout(updatemenus=[])
            with out:
                fig.show()

        for w in (kind, picks, strength):
            w.observe(redraw, names="value")
        display(widgets.VBox([kind, widgets.HBox([picks, strength]), out]))
        redraw()

    def scree(self, analyzer, threshold=0.90):
        pct = analyzer.ratio * 100
        cum = analyzer.cumulative * 100
        k = analyzer.components_for(threshold)
        colors = [TEAL if i < k else SLATE for i in range(len(pct))]
        fig = go.Figure()
        fig.add_bar(x=analyzer.labels, y=pct, name="Individual", marker_color=colors,
                    text=[f"{v:.1f}%" for v in pct], textposition="outside")
        fig.add_scatter(x=analyzer.labels, y=cum, name="Cumulative", mode="lines+markers+text",
                        line=dict(color=CORAL, width=3, shape="spline"), marker=dict(size=10),
                        text=[f"{v:.1f}%" for v in cum], textposition="top left")
        fig.add_hline(y=threshold * 100, line_dash="dash", line_color=NAVY,
                      annotation_text=f"{threshold:.0%} target", annotation_position="bottom right")
        fig.update_layout(yaxis=dict(title="Variance explained (%)", range=[0, 110]),
                          legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
        return self._finish(fig, f"{k} components are needed to keep {threshold:.0%} of the variance",
                            "Scree plot of PCA on the min-max normalized features")

    def scaling_comparison(self, table, threshold=0.90):
        fig = go.Figure()
        styles = {"Raw units": CORAL, "Min-max [0, 1]": TEAL, "Standardized (z-score)": AMBER}
        for name in table.columns:
            fig.add_scatter(x=table.index, y=table[name] * 100, name=name, mode="lines+markers",
                            line=dict(color=styles.get(name, SLATE), width=3, shape="spline"), marker=dict(size=9),
                            hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>")
        fig.add_hline(y=threshold * 100, line_dash="dash", line_color=NAVY)
        fig.update_layout(yaxis=dict(title="Cumulative variance explained (%)", range=[0, 103]),
                          legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
        return self._finish(fig, "Scaling changes the story PCA tells",
                            "On raw units PC1 looks like it explains 94%, but that is just Lot_Size's large numbers")

    def loadings_heatmap(self, loadings):
        fig = go.Figure(go.Heatmap(z=loadings.values, x=loadings.columns, y=loadings.index, colorscale=DIVERGING,
                                   zmin=-1, zmax=1, text=loadings.round(2).values, texttemplate="%{text}",
                                   hovertemplate="%{y} on %{x}: %{z:.3f}<extra></extra>",
                                   colorbar=dict(title="Loading")))
        fig.update_yaxes(autorange="reversed")
        return self._finish(fig, "Each component is mostly one feature",
                            "Loadings close to ±1 mean the PC is essentially that original feature", height=460)

    def biplot(self, analyzer, price, scale=0.6):
        scores = analyzer.scores
        load = analyzer.loadings()
        fig = go.Figure()
        fig.add_scatter(x=scores["PC1"], y=scores["PC2"], mode="markers",
                        marker=dict(size=6, color=price, colorscale="Tealgrn", opacity=0.75,
                                    colorbar=dict(title="Price ($)"), line=dict(width=0)),
                        customdata=price, hovertemplate="PC1 %{x:.2f}<br>PC2 %{y:.2f}<br>Price $%{customdata:,.0f}<extra></extra>",
                        showlegend=False)
        for feat in load.index:
            x, y = load.loc[feat, "PC1"] * scale, load.loc[feat, "PC2"] * scale
            if abs(x) + abs(y) < 0.05:
                continue
            fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=3, arrowwidth=2.5, arrowcolor=CORAL)
            fig.add_annotation(x=x * 1.12, y=y * 1.12, text=f"<b>{feat}</b>", showarrow=False, font=dict(color=CORAL))
        fig.update_layout(xaxis_title=f"PC1 ({analyzer.ratio[0]:.1%})", yaxis_title=f"PC2 ({analyzer.ratio[1]:.1%})")
        return self._finish(fig, "Houses in PC space, coloured by price",
                            "No colour gradient across the cloud: the top components do not separate cheap from expensive homes",
                            height=560)

    def target_correlation(self, corr):
        colors = [CORAL if abs(v) == corr.abs().max() else SLATE for v in corr]
        fig = go.Figure(go.Bar(x=corr.index, y=corr.values, marker_color=colors,
                               text=[f"{v:+.3f}" for v in corr], textposition="outside"))
        fig.add_hrect(y0=-0.1, y1=0.1, fillcolor=SLATE, opacity=0.08, line_width=0,
                      annotation_text="|r| < 0.1 = negligible", annotation_position="top left")
        fig.update_yaxes(title="Pearson r with Price", range=[-0.15, 0.15])
        return self._finish(fig, "Which component tracks Price?",
                            "The highest-variance PCs are not the ones most related to Price - and none are strongly related")

    def importance_lenses(self, variance_share, impact):
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                            subplot_titles=("Share of feature-space variance (PCA)", "|Correlation| with Price"))
        v = variance_share.sort_values()
        fig.add_bar(y=v.index, x=v.values * 100, orientation="h", marker_color=TEAL,
                    text=[f"{x:.1f}%" for x in v.values * 100], textposition="outside", showlegend=False, row=1, col=1)
        c = impact["Correlation r"].abs().sort_values()
        fig.add_bar(y=c.index, x=c.values, orientation="h", marker_color=CORAL,
                    text=[f"{x:.3f}" for x in c.values], textposition="outside", showlegend=False, row=1, col=2)
        fig.update_xaxes(range=[0, v.max() * 130], row=1, col=1)
        fig.update_xaxes(range=[0, c.max() * 1.4], row=1, col=2)
        return self._finish(fig, "Two meanings of 'important'",
                            "What drives variation in the features is not the same as what drives Price", height=440)

    def price_impact(self, impact, r2):
        t = impact.sort_values("Impact ($, min to max)")
        colors = [TEAL if v > 0 else CORAL for v in t["Impact ($, min to max)"]]
        fig = go.Figure(go.Scatter(
            x=t["Impact ($, min to max)"], y=t.index, mode="markers",
            marker=dict(size=16, color=colors, line=dict(color="white", width=2)),
            error_x=dict(type="data", symmetric=False, color=SLATE, thickness=2, width=8,
                         array=t["CI high"] - t["Impact ($, min to max)"],
                         arrayminus=t["Impact ($, min to max)"] - t["CI low"]),
            hovertemplate="%{y}<br>Impact: $%{x:,.0f}<extra></extra>"))
        fig.add_vline(x=0, line_dash="dash", line_color=NAVY)
        fig.update_xaxes(title="Change in predicted Price moving the feature from its min to its max ($)", tickformat="$,.0f")
        return self._finish(fig, "Feature impact on Price, with 95% bootstrap intervals",
                            f"Every interval crosses $0 and R² = {r2:.4f}: no feature reliably moves Price in this dataset",
                            height=440)
