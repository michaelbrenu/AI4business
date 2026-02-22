"""Automatic data profiling and discrepancy detection for uploaded CSVs."""

import pandas as pd
import numpy as np


def profile_dataset(df):
    """Generate a comprehensive data quality profile."""
    profile = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "issues": [],
        "summary": {},
    }

    # Overall stats
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())
    profile["summary"] = {
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_pct": round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
        "duplicate_rows": total_duplicates,
        "duplicate_pct": round(total_duplicates / len(df) * 100, 2) if len(df) > 0 else 0,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }

    for col in df.columns:
        col_info = _profile_column(df, col)
        profile["columns"][col] = col_info

        # Collect issues for this column
        for issue in col_info.get("issues", []):
            profile["issues"].append({"column": col, **issue})

    return profile


def _profile_column(df, col):
    """Profile a single column and detect issues."""
    series = df[col]
    info = {
        "dtype": str(series.dtype),
        "non_null": int(series.count()),
        "null_count": int(series.isnull().sum()),
        "null_pct": round(float(series.isnull().mean() * 100), 2),
        "unique_count": int(series.nunique()),
        "unique_pct": round(float(series.nunique() / len(series) * 100), 2) if len(series) > 0 else 0,
        "sample_values": series.dropna().head(5).tolist(),
        "issues": [],
    }

    # Detect data type
    inferred_type = _infer_semantic_type(series)
    info["inferred_type"] = inferred_type

    # Missing values issue
    if info["null_pct"] > 0:
        severity = "high" if info["null_pct"] > 20 else "medium" if info["null_pct"] > 5 else "low"
        info["issues"].append({
            "type": "missing_values",
            "severity": severity,
            "description": f"{info['null_count']} missing values ({info['null_pct']}%)",
            "fix": f"Impute with {'median' if inferred_type == 'numeric' else 'mode'} or drop rows",
        })

    # Numeric-specific checks
    if inferred_type == "numeric":
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) > 0:
            info["stats"] = {
                "mean": round(float(numeric_series.mean()), 2),
                "median": round(float(numeric_series.median()), 2),
                "std": round(float(numeric_series.std()), 2),
                "min": round(float(numeric_series.min()), 2),
                "max": round(float(numeric_series.max()), 2),
                "q1": round(float(numeric_series.quantile(0.25)), 2),
                "q3": round(float(numeric_series.quantile(0.75)), 2),
            }

            # Outlier detection (IQR method) — skip binary columns
            q1, q3 = numeric_series.quantile(0.25), numeric_series.quantile(0.75)
            iqr = q3 - q1
            is_binary = set(numeric_series.unique()).issubset({0, 1, 0.0, 1.0})
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((numeric_series < lower) | (numeric_series > upper)).sum()
            if outliers > 0 and not is_binary and iqr > 0:
                info["stats"]["outlier_count"] = int(outliers)
                info["issues"].append({
                    "type": "outliers",
                    "severity": "medium" if outliers / len(numeric_series) < 0.05 else "high",
                    "description": f"{int(outliers)} potential outliers detected (IQR method, range: {lower:.1f} to {upper:.1f})",
                    "fix": "Clip to IQR bounds or investigate individually",
                })

            # Negative values check for typically positive columns
            if numeric_series.min() < 0:
                info["issues"].append({
                    "type": "negative_values",
                    "severity": "low",
                    "description": f"Contains negative values (min: {numeric_series.min():.2f})",
                    "fix": "Verify if negative values are expected for this column",
                })

    # Categorical-specific checks
    elif inferred_type == "categorical":
        value_counts = series.dropna().value_counts()
        info["top_values"] = value_counts.head(10).to_dict()

        # High cardinality warning
        if info["unique_count"] > 50 and info["unique_pct"] > 50:
            info["issues"].append({
                "type": "high_cardinality",
                "severity": "medium",
                "description": f"{info['unique_count']} unique values - may be an ID column or need grouping",
                "fix": "Consider if this is an identifier column or if values should be grouped",
            })

        # Whitespace/casing inconsistencies
        str_values = series.dropna().astype(str)
        stripped = str_values.str.strip()
        whitespace_issues = (str_values != stripped).sum()
        if whitespace_issues > 0:
            info["issues"].append({
                "type": "whitespace",
                "severity": "low",
                "description": f"{int(whitespace_issues)} values have leading/trailing whitespace",
                "fix": "Strip whitespace from values",
            })

        # Case inconsistencies
        lowered = stripped.str.lower()
        if lowered.nunique() < stripped.nunique():
            info["issues"].append({
                "type": "case_inconsistency",
                "severity": "medium",
                "description": "Possible case inconsistencies (e.g., 'Male' vs 'male')",
                "fix": "Standardize text casing",
            })

    # Constant column
    if info["unique_count"] <= 1 and info["non_null"] > 0:
        info["issues"].append({
            "type": "constant",
            "severity": "medium",
            "description": "Column has only one unique value - provides no analytical value",
            "fix": "Consider removing this column",
        })

    return info


def _infer_semantic_type(series):
    """Infer the semantic type of a column."""
    if series.dtype in ["int64", "float64", "int32", "float32"]:
        return "numeric"

    # Try to parse as numeric
    try:
        numeric = pd.to_numeric(series.dropna(), errors="coerce")
        if numeric.notna().mean() > 0.8:
            return "numeric"
    except Exception:
        pass

    # Try to parse as datetime
    try:
        if series.dropna().astype(str).str.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > 0.8:
            return "datetime"
    except Exception:
        pass

    # Check if boolean-like
    unique_lower = set(series.dropna().astype(str).str.lower().unique())
    if unique_lower.issubset({"true", "false", "0", "1", "yes", "no", "y", "n"}):
        return "boolean"

    return "categorical"


def suggest_column_mapping(df):
    """Suggest which columns map to expected analytical roles."""
    suggestions = {
        "id_column": None,
        "target_column": None,
        "numeric_columns": [],
        "categorical_columns": [],
        "date_column": None,
    }

    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        series = df[col]
        inferred = _infer_semantic_type(series)

        # ID detection
        if any(kw in col_lower for kw in ["id", "identifier", "key", "index"]):
            if suggestions["id_column"] is None:
                suggestions["id_column"] = col
                continue

        # Target/outcome detection
        if any(kw in col_lower for kw in ["pass", "fail", "target", "outcome", "result", "label", "grade", "score", "status"]):
            if suggestions["target_column"] is None:
                suggestions["target_column"] = col

        # Date detection
        if inferred == "datetime" or any(kw in col_lower for kw in ["date", "time", "year", "semester", "quarter", "month", "period"]):
            if suggestions["date_column"] is None:
                suggestions["date_column"] = col

        # Numeric vs categorical
        if inferred == "numeric":
            suggestions["numeric_columns"].append(col)
        elif inferred in ["categorical", "boolean"]:
            suggestions["categorical_columns"].append(col)

    return suggestions


def generate_cleaning_plan(profile):
    """Generate an ordered list of recommended cleaning actions based on the profile."""
    actions = []

    # Duplicates first
    if profile["summary"]["duplicate_rows"] > 0:
        actions.append({
            "action": "remove_duplicates",
            "description": f"Remove {profile['summary']['duplicate_rows']} duplicate rows",
            "impact": f"Reduces dataset from {profile['shape']['rows']} to {profile['shape']['rows'] - profile['summary']['duplicate_rows']} rows",
            "auto": True,
        })

    # Column-level fixes
    for col, info in profile["columns"].items():
        for issue in info.get("issues", []):
            if issue["type"] == "whitespace":
                actions.append({
                    "action": "strip_whitespace",
                    "column": col,
                    "description": f"Strip whitespace from '{col}'",
                    "impact": issue["description"],
                    "auto": True,
                })

            elif issue["type"] == "case_inconsistency":
                actions.append({
                    "action": "standardize_case",
                    "column": col,
                    "description": f"Standardize casing in '{col}' (title case)",
                    "impact": issue["description"],
                    "auto": True,
                })

            elif issue["type"] == "missing_values":
                fill_method = "median" if info["inferred_type"] == "numeric" else "mode"
                actions.append({
                    "action": "impute_missing",
                    "column": col,
                    "method": fill_method,
                    "description": f"Fill {info['null_count']} missing values in '{col}' with {fill_method}",
                    "impact": issue["description"],
                    "auto": True,
                })

            elif issue["type"] == "outliers":
                actions.append({
                    "action": "clip_outliers",
                    "column": col,
                    "description": f"Clip outliers in '{col}' to IQR bounds",
                    "impact": issue["description"],
                    "auto": False,
                })

            elif issue["type"] == "constant":
                actions.append({
                    "action": "drop_column",
                    "column": col,
                    "description": f"Drop constant column '{col}'",
                    "impact": issue["description"],
                    "auto": False,
                })

    return actions
