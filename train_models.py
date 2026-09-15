#!/usr/bin/env python3
"""
train_models.py - Multi-Model Training Pipeline for Customer Churn Predictor
Includes clean 34-feature preprocessing, SMOTE oversampling, multi-model evaluation,
and feature direction/attribution metadata for explainability.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

FRIENDLY_FEATURE_NAMES = {
    "tenure": "Account Tenure (Months)",
    "PhoneService": "Phone Service Enabled",
    "PaperlessBilling": "Paperless Billing",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
    "gender": "Gender (Male)",
    "SeniorCitizen": "Senior Citizen Status",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "MultipleLines_No phone service": "No Phone Service",
    "MultipleLines_Yes": "Multiple Phone Lines",
    "InternetService_Fiber optic": "Fiber Optic Internet",
    "InternetService_No": "No Internet Service",
    "OnlineSecurity_No internet service": "No Internet Security",
    "OnlineSecurity_Yes": "Online Security Active",
    "OnlineBackup_No internet service": "No Internet Backup",
    "OnlineBackup_Yes": "Online Backup Active",
    "DeviceProtection_No internet service": "No Device Protection",
    "DeviceProtection_Yes": "Device Protection Active",
    "TechSupport_No internet service": "No Tech Support Service",
    "TechSupport_Yes": "Tech Support Active",
    "StreamingTV_No internet service": "No Streaming TV Service",
    "StreamingTV_Yes": "Streaming TV Active",
    "StreamingMovies_No internet service": "No Streaming Movies Service",
    "StreamingMovies_Yes": "Streaming Movies Active",
    "Contract_One year": "One-Year Contract",
    "Contract_Two year": "Two-Year Contract",
    "PaymentMethod_Credit card (automatic)": "Credit Card (Auto-Pay)",
    "PaymentMethod_Electronic check": "Electronic Check Payment",
    "PaymentMethod_Mailed check": "Mailed Check Payment",
    "tenure_group_12-24 Months": "Tenure Cohort (1-2 Years)",
    "tenure_group_24-48 Months": "Tenure Cohort (2-4 Years)",
    "tenure_group_48-60 Months": "Tenure Cohort (4-5 Years)",
    "tenure_group_60+ Months": "Tenure Cohort (5+ Years)",
}

def load_and_merge_data():
    """Load and merge the three telecom churn dataset files."""
    churn_path = BASE_DIR / "churn_data.csv"
    customer_path = BASE_DIR / "customer_data.csv"
    internet_path = BASE_DIR / "internet_data.csv"

    if not (churn_path.exists() and customer_path.exists() and internet_path.exists()):
        raise FileNotFoundError("One or more CSV dataset files are missing in project directory.")

    print("Loading datasets...")
    churn_df = pd.read_csv(churn_path)
    customer_df = pd.read_csv(customer_path)
    internet_df = pd.read_csv(internet_path)

    df = churn_df.merge(customer_df, on="customerID").merge(internet_df, on="customerID")
    print(f"Merged dataset shape: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame):
    """Clean, engineer features, and encode variables without customerID leakage."""
    df = df.copy()

    # Clean numeric fields
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Encode target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Explicitly drop customerID first
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Tenure grouping feature engineering
    def tenure_group(tenure):
        if tenure <= 12:
            return "0-12 Months"
        elif tenure <= 24:
            return "12-24 Months"
        elif tenure <= 48:
            return "24-48 Months"
        elif tenure <= 60:
            return "48-60 Months"
        else:
            return "60+ Months"

    df["tenure_group"] = df["tenure"].apply(tenure_group)

    # Binary columns encoding
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    # Multi-category columns for one-hot encoding
    cat_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
        "tenure_group",
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Convert all boolean columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"Clean features count: {X.shape[1]}")
    return X, y

def train_and_evaluate(X, y):
    """Train multiple models with SMOTE class-balancing and evaluate on test holdout."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_names = list(X.columns)

    # Compute feature correlations with churn for directional attribution
    feature_directions = {col: float(val) for col, val in X.corrwith(y).items()}

    # Standard scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to training set only
    print("Applying SMOTE oversampling to training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Resampled training set shape: {X_train_res.shape}")

    # Model dictionary
    candidate_models = {
        "Gradient Boosting": (
            GradientBoostingClassifier(
                random_state=42, n_estimators=120, learning_rate=0.08, max_depth=3
            ),
            "Champion ensemble model balancing high AUC-ROC and recall.",
        ),
        "Logistic Regression": (
            LogisticRegression(random_state=42, max_iter=1000, C=1.0, solver="lbfgs"),
            "High sensitivity model optimized for catching maximum churners.",
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=42, n_estimators=150, max_depth=10, min_samples_split=5
            ),
            "Robust tree-based ensemble capturing complex feature interactions.",
        ),
    }

    trained_bundle = {
        "models": {},
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_importances": {},
        "feature_directions": feature_directions,
        "friendly_names": FRIENDLY_FEATURE_NAMES,
    }

    print("\n--- Model Training & Evaluation Benchmark ---")
    for name, (model, desc) in candidate_models.items():
        print(f"Training {name}...")
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        rec = float(recall_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        metrics = {
            "Accuracy": acc,
            "AUC-ROC": auc,
            "Recall": rec,
            "Precision": prec,
            "F1-Score": f1,
        }

        # Extract feature importances or coefficients
        if hasattr(model, "feature_importances_"):
            raw_imp = [float(v) for v in model.feature_importances_]
            # Normalize to 0-1
            total_imp = sum(raw_imp) or 1.0
            norm_imp = [v / total_imp for v in raw_imp]
            importances = dict(zip(feature_names, norm_imp))
        elif hasattr(model, "coef_"):
            raw_coef = [abs(float(v)) for v in model.coef_[0]]
            total_coef = sum(raw_coef) or 1.0
            norm_imp = [v / total_coef for v in raw_coef]
            importances = dict(zip(feature_names, norm_imp))
        else:
            importances = {}

        trained_bundle["models"][name] = {
            "model": model,
            "metrics": metrics,
            "description": desc,
        }
        trained_bundle["feature_importances"][name] = importances

        print(
            f"  {name:20s} | Acc: {acc:.4f} | AUC: {auc:.4f} | Recall: {rec:.4f} | Prec: {prec:.4f} | F1: {f1:.4f}"
        )

    return trained_bundle

def save_artifacts(bundle):
    """Save models bundle and backward-compatible legacy artifacts."""
    bundle_path = BASE_DIR / "models.pkl"
    legacy_model_path = BASE_DIR / "churn_model.pkl"
    legacy_scaler_path = BASE_DIR / "scaler.pkl"

    print(f"\nSaving multi-model bundle to {bundle_path}...")
    joblib.dump(bundle, bundle_path)

    # Save Gradient Boosting as default champion model for legacy churn_model.pkl
    champion = bundle["models"]["Gradient Boosting"]["model"]
    print(f"Updating legacy {legacy_model_path} with champion Gradient Boosting...")
    joblib.dump(champion, legacy_model_path)

    print(f"Updating legacy {legacy_scaler_path} with 34-feature scaler...")
    joblib.dump(bundle["scaler"], legacy_scaler_path)

    print("All artifacts saved successfully!")

def main():
    df = load_and_merge_data()
    X, y = preprocess_data(df)
    bundle = train_and_evaluate(X, y)
    save_artifacts(bundle)
    print("\nPhase 2 ML training and attribution bundle generated successfully.")

if __name__ == "__main__":
    main()
