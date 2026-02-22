"""Dynamic Plotly visualization functions that work with any mapped columns."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.utils import COLORS


def chart_target_distribution(df, target_col, title=None):
    """Histogram of target column values."""
    title = title or f"Distribution of {target_col.replace('_', ' ').title()}"

    numeric = pd.to_numeric(df[target_col], errors="coerce")
    if numeric.notna().mean() > 0.8:
        fig = px.histogram(
            df, x=target_col, nbins=25,
            color_discrete_sequence=[COLORS["primary"]],
            title=title,
        )
        fig.update_layout(
            xaxis_title=target_col.replace("_", " ").title(),
            yaxis_title="Count",
            template="plotly_white", height=450,
        )
    else:
        counts = df[target_col].value_counts().reset_index()
        counts.columns = [target_col, "count"]
        fig = px.bar(
            counts, x=target_col, y="count",
            color_discrete_sequence=[COLORS["primary"]],
            title=title,
        )
        fig.update_layout(template="plotly_white", height=450)
    return fig


def chart_correlation_scatter(df, x_col, y_col, color_col=None, size_col=None, title=None):
    """Scatter plot between two numeric columns with optional trendline."""
    title = title or f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}"

    plot_df = df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    kwargs = {
        "x": x_col, "y": y_col,
        "opacity": 0.6, "trendline": "ols",
        "title": title,
    }

    if color_col and color_col in plot_df.columns:
        plot_df[color_col] = plot_df[color_col].astype(str)
        kwargs["color"] = color_col

    if size_col and size_col in plot_df.columns:
        plot_df[size_col] = pd.to_numeric(plot_df[size_col], errors="coerce")
        kwargs["size"] = size_col
        kwargs["size_max"] = 12

    fig = px.scatter(plot_df, **kwargs)

    # Correlation annotation
    corr = plot_df[x_col].corr(plot_df[y_col])
    if not pd.isna(corr):
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"r = {corr:.3f}", showarrow=False,
            font=dict(size=14, color=COLORS["primary"]),
            bgcolor="white", bordercolor=COLORS["primary"],
            borderwidth=1, borderpad=4,
        )

    fig.update_layout(template="plotly_white", height=450)
    return fig


def chart_boxplot_by_category(df, numeric_col, category_col, title=None):
    """Box plot: numeric values grouped by a categorical column."""
    title = title or f"{numeric_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}"

    plot_df = df.copy()
    plot_df[numeric_col] = pd.to_numeric(plot_df[numeric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[numeric_col])

    fig = px.box(
        plot_df, x=category_col, y=numeric_col,
        color=category_col, title=title, points="outliers",
    )

    # Mean markers
    means = plot_df.groupby(category_col)[numeric_col].mean()
    categories = plot_df[category_col].unique()
    fig.add_trace(go.Scatter(
        x=list(means.index), y=list(means.values),
        mode="markers", name="Mean",
        marker=dict(symbol="diamond", size=12, color="black",
                    line=dict(width=2, color="white")),
    ))

    fig.update_layout(template="plotly_white", height=450, showlegend=True)
    return fig


def chart_category_breakdown(df, category_col, target_col, title=None):
    """Bar chart showing target metric breakdown by category."""
    title = title or f"{target_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}"

    numeric_target = pd.to_numeric(df[target_col], errors="coerce")
    is_binary = set(numeric_target.dropna().unique()).issubset({0, 1, 0.0, 1.0})

    if is_binary:
        # Pass/fail style grouped bar — use numeric-converted target
        tmp = df[[category_col]].copy()
        tmp["_target_num"] = numeric_target
        grouped = tmp.groupby(category_col)["_target_num"].agg(["sum", "count"]).reset_index()
        grouped.columns = [category_col, "positive", "total"]
        grouped["positive_rate"] = (grouped["positive"] / grouped["total"] * 100).round(1)
        grouped["negative_rate"] = (100 - grouped["positive_rate"]).round(1)
        overall_rate = numeric_target.mean() * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=grouped[category_col], y=grouped["positive_rate"],
            name="Positive", marker_color=COLORS["success"],
            text=grouped["positive_rate"].apply(lambda x: f"{x:.1f}%"), textposition="auto",
        ))
        fig.add_trace(go.Bar(
            x=grouped[category_col], y=grouped["negative_rate"],
            name="Negative", marker_color=COLORS["danger"],
            text=grouped["negative_rate"].apply(lambda x: f"{x:.1f}%"), textposition="auto",
        ))
        fig.add_hline(y=overall_rate, line_dash="dash", line_color=COLORS["primary"],
                      annotation_text=f"Overall: {overall_rate:.1f}%")
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 105]))
    else:
        # Mean value bar chart — use numeric-converted target
        tmp = df[[category_col]].copy()
        tmp["_target_num"] = numeric_target
        grouped = tmp.groupby(category_col)["_target_num"].agg(["mean", "count"]).reset_index()
        grouped.columns = [category_col, "mean_value", "count"]
        grouped["mean_value"] = grouped["mean_value"].round(2)

        fig = px.bar(
            grouped, x=category_col, y="mean_value",
            text="mean_value", color_discrete_sequence=[COLORS["primary"]],
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="auto")

    fig.update_layout(
        title=title,
        xaxis_title=category_col.replace("_", " ").title(),
        yaxis_title="Percentage (%)" if is_binary else target_col.replace("_", " ").title(),
        template="plotly_white", height=450,
    )
    return fig


def chart_trend_over_time(df, time_col, value_col, group_col=None, title=None):
    """Line chart showing trends over a time/period column."""
    title = title or f"{value_col.replace('_', ' ').title()} Trend over {time_col.replace('_', ' ').title()}"

    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")

    if group_col and group_col in plot_df.columns:
        trend = plot_df.groupby([time_col, group_col])[value_col].mean().reset_index()
        fig = px.line(
            trend, x=time_col, y=value_col, color=group_col,
            markers=True, title=title,
        )

        # Overall average line
        overall = plot_df.groupby(time_col)[value_col].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=overall[time_col], y=overall[value_col],
            mode="lines+markers", name="Overall Average",
            line=dict(dash="dash", color="black", width=2),
            marker=dict(size=8),
        ))
    else:
        trend = plot_df.groupby(time_col)[value_col].mean().reset_index()
        fig = px.line(trend, x=time_col, y=value_col, markers=True, title=title)

    fig.update_layout(template="plotly_white", height=450)
    return fig


def chart_correlation_heatmap(df, numeric_columns, title="Correlation Heatmap"):
    """Heatmap of correlations between numeric columns."""
    plot_df = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    corr_matrix = plot_df.corr().round(3)

    fig = px.imshow(
        corr_matrix, text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title=title,
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig
