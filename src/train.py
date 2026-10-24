
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "amazon_employee_eda_model_assignment.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "return_model.joblib"


def main():
    df = pd.read_csv(DATA_PATH, engine="c").drop_duplicates().copy()

    target = "returned_flag"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df = df.dropna(subset=[target])

    # Exclude target, direct outcome columns and identifiers.
    # Keep other usable features, including review-related features.
    drop_cols = [
        "returned_flag",
        "returned",
        "order_status",
        "order_id",
        "customer_id",
        "product_id",
        "order_date",
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target].astype(int)

    # Exclude columns that contain no usable information.
    usable_cols = [
        col for col in X.columns
        if not X[col].isna().all()
    ]
    X = X[usable_cols]

    # Avoid accidental non-numeric values in the target.
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("returned_flag must contain only 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=10,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Random Forest Evaluation ===")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    ap = average_precision_score(y_test, y_prob)
    print(f"Average Precision: {ap:.4f}")
    print(f"Test-set return rate: {y_test.mean():.2%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    print(
        "Precision-Recall AUC (trapezoidal): "
        f"{auc(recall[::-1], precision[::-1]):.4f}"
    )

    # Feature importance for the fitted Random Forest.
    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.named_steps[
            "classifier"
        ].feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 15 transformed features:")
    print(importance.head(15).to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model and metadata together.
    artifact = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target,
        "average_precision": float(ap),
        "test_return_rate": float(y_test.mean()),
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
