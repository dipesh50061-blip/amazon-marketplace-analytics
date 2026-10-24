
from pathlib import Path

import joblib
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "return_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run src/train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_return(order_data: dict):
    model = load_model()

    order_df = pd.DataFrame([order_data])

    prediction = int(model.predict(order_df)[0])
    probability = float(model.predict_proba(order_df)[0, 1])

    return {
        "prediction": prediction,
        "label": "Returned" if prediction == 1 else "Not Returned",
        "return_probability": round(probability, 4),
    }



if __name__ == "__main__":
    DATA_PATH = (
        PROJECT_ROOT
        / "data"
        / "amazon_employee_eda_model_assignment.csv"
    )

    df = pd.read_csv(DATA_PATH).drop_duplicates()

    drop_cols = [
        "returned_flag",
        "returned",
        "order_status",
        "review_text",
        "rating",
        "delivery_days",
        "order_id",
        "customer_id",
        "product_id",
        "order_date",
    ]

    # Select one example and keep only training features
    X_sample = df.drop(columns=drop_cols, errors="ignore").iloc[[0]]

    model = load_model()

    prediction = int(model.predict(X_sample)[0])
    probability = float(model.predict_proba(X_sample)[0, 1])

    print("Prediction:", "Returned" if prediction == 1 else "Not Returned")
    print(f"Predicted return probability: {probability:.2%}")
    print("Actual label:", int(df["returned_flag"].iloc[0]))

