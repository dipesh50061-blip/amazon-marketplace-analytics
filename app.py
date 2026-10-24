
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "return_model.joblib"
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "amazon_employee_eda_model_assignment.csv"
)

st.set_page_config(
    page_title="Marketplace Return Predictor",
    page_icon="📦",
    layout="wide",
)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH, engine="c").drop_duplicates()


st.title("Marketplace Return Prediction")
st.write(
    "Enter order details to estimate the model's predicted "
    "return probability."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset file was not found in the data folder.")
    st.stop()

artifact = load_artifact()
model = artifact["model"]
feature_columns = artifact["feature_columns"]
numeric_features = set(artifact["numeric_features"])
categorical_features = set(artifact["categorical_features"])

reference_df = load_reference_data()
reference_df = reference_df.dropna(
    subset=["returned_flag"]
)

reference_X = reference_df.drop(
    columns=[
        "returned_flag",
        "returned",
        "order_status",
        "order_id",
        "customer_id",
        "product_id",
        "order_date",
    ],
    errors="ignore",
)

# Use the exact columns saved with the model.
reference_X = reference_X.reindex(columns=feature_columns)

st.caption(
    "This is a model demonstration using synthetic assignment data. "
    "Predictions are not guaranteed to be calibrated real-world risks."
)

with st.form("prediction_form"):
    st.subheader("Order details")

    user_values = {}

    col1, col2 = st.columns(2)

    for index, feature in enumerate(feature_columns):
        container = col1 if index % 2 == 0 else col2

        with container:
            series = reference_X[feature]

            if feature in numeric_features:
                values = pd.to_numeric(
                    series, errors="coerce"
                ).dropna()

                default_value = (
                    float(values.median()) if not values.empty else 0.0
                )

                min_value = (
                    float(values.min()) if not values.empty else 0.0
                )
                max_value = (
                    float(values.max()) if not values.empty else 100.0
                )

                if min_value == max_value:
                    max_value = min_value + 1.0

                user_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=min_value,
                    max_value=max_value,
                    value=min(max(default_value, min_value), max_value),
                    key=f"num_{feature}",
                )

            else:
                options = (
                    series.dropna()
                    .astype(str)
                    .value_counts()
                    .index.tolist()
                )

                if not options:
                    options = ["Unknown"]

                user_values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options=options,
                    key=f"cat_{feature}",
                )

    submitted = st.form_submit_button(
        "Predict Return",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_df = pd.DataFrame(
        [user_values],
        columns=feature_columns,
    )

    # Restore numeric types for numeric model columns.
    for feature in numeric_features:
        input_df[feature] = pd.to_numeric(
            input_df[feature], errors="coerce"
        )

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    st.divider()
    st.subheader("Prediction result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.warning("Predicted result: Returned")
        else:
            st.success("Predicted result: Not Returned")

    with result_col2:
        st.metric(
            "Model-estimated return probability",
            f"{probability:.2%}",
        )

    st.progress(probability)

    st.caption(
        "This probability is the model's output, not a guarantee. "
        "The dataset is synthetic and the model has shown limited "
        "predictive improvement over the return-rate baseline."
    )
