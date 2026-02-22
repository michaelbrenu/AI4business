"""Dynamic data cleaning based on profiler results and user selections."""

import pandas as pd
import numpy as np


def apply_cleaning_actions(df, actions):
    """Apply a list of cleaning actions to the DataFrame. Return (cleaned_df, log)."""
    df = df.copy()
    log = []

    for action in actions:
        act_type = action["action"]
        col = action.get("column")

        if act_type == "remove_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            log.append(f"Removed {removed} duplicate rows")

        elif act_type == "strip_whitespace" and col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                log.append(f"Stripped whitespace from '{col}'")

        elif act_type == "standardize_case" and col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.title()
                log.append(f"Standardized casing in '{col}' to title case")

        elif act_type == "impute_missing" and col in df.columns:
            method = action.get("method", "median")
            null_count = df[col].isnull().sum()
            if null_count > 0:
                if method == "median":
                    fill_val = pd.to_numeric(df[col], errors="coerce").median()
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
                elif method == "mean":
                    fill_val = pd.to_numeric(df[col], errors="coerce").mean()
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
                elif method == "mode":
                    mode_val = df[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                    df[col] = df[col].fillna(fill_val)
                elif method == "drop":
                    df = df.dropna(subset=[col])
                    fill_val = "N/A (rows dropped)"
                else:
                    fill_val = method  # custom value
                    df[col] = df[col].fillna(fill_val)
                log.append(f"Filled {null_count} missing values in '{col}' with {method} ({fill_val})")

        elif act_type == "clip_outliers" and col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = numeric_col.quantile(0.25), numeric_col.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            clipped = numeric_col.clip(lower, upper)
            changed = (numeric_col != clipped).sum()
            df[col] = clipped
            log.append(f"Clipped {int(changed)} outlier values in '{col}' to [{lower:.2f}, {upper:.2f}]")

        elif act_type == "drop_column" and col in df.columns:
            df = df.drop(columns=[col])
            log.append(f"Dropped column '{col}'")

        elif act_type == "convert_numeric" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            log.append(f"Converted '{col}' to numeric type")

    return df, log


def get_dynamic_filter_options(df, categorical_columns, numeric_columns):
    """Return filter options based on actual column content."""
    options = {}

    for col in categorical_columns:
        if col in df.columns:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            options[col] = unique_vals

    for col in numeric_columns:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(numeric_series) > 0:
                options[col] = {
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                }

    return options


def apply_dynamic_filters(df, filters, categorical_columns, numeric_columns):
    """Apply user-selected filters dynamically."""
    filtered = df.copy()

    for col in categorical_columns:
        if col in filters and filters[col]:
            filtered = filtered[filtered[col].isin(filters[col])]

    for col in numeric_columns:
        key = f"{col}_range"
        if key in filters:
            low, high = filters[key]
            numeric_col = pd.to_numeric(filtered[col], errors="coerce")
            filtered = filtered[(numeric_col >= low) & (numeric_col <= high)]

    return filtered
