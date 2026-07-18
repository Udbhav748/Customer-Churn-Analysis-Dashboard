"""Train and evaluate churn-prediction models; saves the best pipeline + metrics.

Run: python ml/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_training_data

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="No Service")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def get_feature_importances(pipeline, top_n=15):
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []

    order = np.argsort(importances)[::-1][:top_n]
    return [
        {"feature": feature_names[i].split("__", 1)[-1], "importance": float(importances[i])}
        for i in order
    ]


def main():
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    pipelines = {}
    for name, model in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        pipelines[name] = pipeline

        print(f"\n{name}:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}")
        print(f"  confusion_matrix: {metrics['confusion_matrix']}")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipeline = pipelines[best_name]
    print(f"\nBest model: {best_name} (roc_auc={results[best_name]['roc_auc']:.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "churn_model.joblib")

    metrics_report = {
        "best_model": best_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate_train": float(y_train.mean()),
        "results": results,
        "feature_importances": get_feature_importances(best_pipeline),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR / 'churn_model.joblib'}")
    print(f"Saved metrics to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
