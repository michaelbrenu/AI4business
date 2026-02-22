"""Dynamic predictive model that works with user-mapped columns."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train_model(df, target_col, numeric_features, categorical_features):
    """
    Train logistic regression on dynamically selected features.
    Returns (pipeline, metrics, feature_importances, feature_config).
    """
    all_features = numeric_features + categorical_features
    df_model = df[all_features + [target_col]].copy()

    # Coerce target to binary if not already
    target = pd.to_numeric(df_model[target_col], errors="coerce")
    unique_vals = target.dropna().unique()

    if len(unique_vals) > 2:
        # Convert to binary: above/below median
        median_val = target.median()
        df_model["_target"] = (target >= median_val).astype(int)
        target_labels = [f"Below {median_val:.1f}", f"At/Above {median_val:.1f}"]
    elif set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        df_model["_target"] = target.astype(int)
        target_labels = ["Negative (0)", "Positive (1)"]
    else:
        vals = sorted(unique_vals)
        mapping = {v: i for i, v in enumerate(vals)}
        df_model["_target"] = target.map(mapping).astype(int)
        target_labels = [str(v) for v in vals]

    # Drop NaN
    df_model = df_model.dropna()

    if len(df_model) < 20:
        return None, {"error": "Not enough data after cleaning (need at least 20 rows)"}, {}, {}

    X = df_model[all_features].copy()
    y = df_model["_target"].copy()

    # Coerce numeric
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Handle categorical as string
    for col in categorical_features:
        X[col] = X[col].astype(str)

    X = X.dropna()
    y = y.loc[X.index]

    if len(y.unique()) < 2:
        return None, {"error": "Target variable has only one class after cleaning"}, {}, {}

    # Split
    test_size = min(0.2, max(0.1, 10 / len(X)))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Build preprocessing
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features))

    preprocessor = ColumnTransformer(transformers)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "target_labels": target_labels,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    # Feature importance
    feature_names = list(numeric_features)
    if categorical_features:
        cat_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
        feature_names += cat_names
    coefs = pipeline.named_steps["classifier"].coef_[0]
    importances = dict(zip(feature_names, np.round(coefs, 3).tolist()))

    feature_config = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_col": target_col,
        "all_features": all_features,
    }

    return pipeline, metrics, importances, feature_config


def predict_single(pipeline, input_data, numeric_features, categorical_features):
    """Predict and return (prediction, probability) for a single input."""
    df = pd.DataFrame([input_data])
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_features:
        df[col] = df[col].astype(str)

    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0]
    return int(prediction), float(max(probability))
