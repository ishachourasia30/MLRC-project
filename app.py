import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "road_incident_risk_model.pkl"

st.set_page_config(
    page_title="Road Accident Risk Classification",
    page_icon="🚦",
    layout="centered"
)

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_FILE)

try:
    artifact = load_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
except FileNotFoundError:
    st.error("Model file not found. Run `python train_model.py` first and upload the generated .pkl file.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.title("🚦 Road Accident Risk Classification")
st.write(
    "Enter road, vehicle, and environmental conditions to get a model-predicted risk class."
)
st.info(
    "This is an educational ML prototype. Its predictions are not a substitute for official road-safety guidance."
)

# Dataset is used only to create sensible input ranges and dropdown choices.
try:
    df = pd.read_csv("kaggle_train.csv")
except FileNotFoundError:
    df = None

st.subheader("Enter conditions")
input_values = {}

with st.form("risk_prediction_form"):
    left, right = st.columns(2)
    for i, col in enumerate(feature_columns):
        area = left if i % 2 == 0 else right
        series = df[col] if df is not None and col in df.columns else None

        with area:
            if series is not None and pd.api.types.is_numeric_dtype(series):
                clean = pd.to_numeric(series, errors="coerce").dropna()
                if len(clean):
                    minimum = float(clean.min())
                    maximum = float(clean.max())
                    default = float(clean.median())
                    if minimum == maximum:
                        input_values[col] = st.number_input(
                            col.replace("_", " ").title(), value=default, key=col
                        )
                    else:
                        input_values[col] = st.number_input(
                            col.replace("_", " ").title(),
                            min_value=minimum,
                            max_value=maximum,
                            value=min(max(default, minimum), maximum),
                            key=col
                        )
                else:
                    input_values[col] = st.number_input(
                        col.replace("_", " ").title(), value=0.0, key=col
                    )
            elif series is not None:
                choices = series.dropna().astype(str).unique().tolist()
                if choices:
                    input_values[col] = st.selectbox(
                        col.replace("_", " ").title(), choices, key=col
                    )
                else:
                    input_values[col] = st.text_input(
                        col.replace("_", " ").title(), key=col
                    )
            else:
                input_values[col] = st.text_input(
                    col.replace("_", " ").title(), key=col
                )

    submitted = st.form_submit_button("Predict Accident Risk", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([input_values], columns=feature_columns)

    # Match categorical input types to the original training data when possible.
    if df is not None:
        for col in feature_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                input_df[col] = input_df[col].astype(str)

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted risk class: **{prediction}**")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.named_steps["classifier"].classes_
            probability_df = pd.DataFrame({
                "Risk class": classes,
                "Model probability (%)": (probabilities * 100).round(2)
            }).sort_values("Model probability (%)", ascending=False)
            st.subheader("Model probabilities")
            st.bar_chart(probability_df.set_index("Risk class"))
            st.dataframe(probability_df, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption("Check that the model file was generated from the same dataset and feature columns.")

with st.expander("About this project"):
    st.write("The model is trained from kaggle_train.csv using a Random Forest classifier and a preprocessing pipeline.")
    st.write("Risk class meanings should be confirmed from the original dataset documentation.")
