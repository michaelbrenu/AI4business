"""Shared constants, color palettes, and helper functions."""

import pandas as pd

COLORS = {
    "primary": "#4361EE",
    "secondary": "#3A0CA3",
    "success": "#06D6A0",
    "danger": "#EF476F",
    "warning": "#FFD166",
    "info": "#118AB2",
}

BAND_COLORS = {
    "A": "#06D6A0",
    "B": "#118AB2",
    "C": "#FFD166",
    "D": "#F77F00",
    "F": "#EF476F",
}

PERFORMANCE_BANDS = {"A": (90, 100), "B": (80, 89), "C": (70, 79), "D": (50, 69), "F": (0, 49)}


def assign_performance_band(grade):
    """Return letter band for a numeric grade."""
    if grade >= 90:
        return "A"
    elif grade >= 80:
        return "B"
    elif grade >= 70:
        return "C"
    elif grade >= 50:
        return "D"
    else:
        return "F"


def format_percentage(value):
    """Format a float as a percentage string."""
    return f"{value:.1f}%"


def compute_dynamic_stats(df, mapping):
    """Compute summary statistics based on user-mapped columns."""
    stats = {"total_records": len(df)}

    id_col = mapping.get("id_column")
    target_col = mapping.get("target_column")
    numeric_cols = mapping.get("numeric_columns", [])
    categorical_cols = mapping.get("categorical_columns", [])
    time_col = mapping.get("date_column")

    if id_col and id_col in df.columns:
        stats["unique_entities"] = int(df[id_col].nunique())

    if target_col and target_col in df.columns:
        target_series = pd.to_numeric(df[target_col], errors="coerce")
        if target_series.notna().any():
            stats["target_mean"] = round(float(target_series.mean()), 2)
            stats["target_median"] = round(float(target_series.median()), 2)
            stats["target_std"] = round(float(target_series.std()), 2)
            stats["target_min"] = round(float(target_series.min()), 2)
            stats["target_max"] = round(float(target_series.max()), 2)

            # Check if binary
            unique_vals = target_series.dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                stats["positive_rate"] = round(float(target_series.mean() * 100), 1)

    # Numeric column stats
    for col in numeric_cols:
        if col in df.columns and col != target_col:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                stats[f"{col}_mean"] = round(float(series.mean()), 2)
                if target_col and target_col in df.columns:
                    target_series = pd.to_numeric(df[target_col], errors="coerce")
                    corr = series.corr(target_series)
                    if not pd.isna(corr):
                        stats[f"{col}_target_corr"] = round(float(corr), 3)

    # Categorical breakdowns
    for col in categorical_cols:
        if col in df.columns and target_col and target_col in df.columns:
            breakdown = df.groupby(col)[target_col].apply(
                lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 3)
            ).to_dict()
            stats[f"{col}_breakdown"] = breakdown

    # Time trends
    if time_col and time_col in df.columns and target_col and target_col in df.columns:
        trend = df.groupby(time_col)[target_col].apply(
            lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 2)
        ).to_dict()
        stats["time_trend"] = trend

    return stats
