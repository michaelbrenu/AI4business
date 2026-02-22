"""Load and validate uploaded CSV datasets."""

import pandas as pd
import streamlit as st
import os
import io


def load_uploaded_file(uploaded_file):
    """Read an uploaded CSV/Excel file and return a DataFrame."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
        return df, None
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def load_sample_data():
    """Load the built-in sample education dataset."""
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "education_data.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


def detect_separator(uploaded_file):
    """Try to detect CSV separator."""
    content = uploaded_file.getvalue().decode("utf-8", errors="replace")
    first_line = content.split("\n")[0]
    for sep in [",", ";", "\t", "|"]:
        if sep in first_line:
            return sep
    return ","
