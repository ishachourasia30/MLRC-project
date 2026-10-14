"""Train a road incident risk classification model and save it for Streamlit."""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = "kaggle_train.csv"
MODEL_FILE = "road_incident_risk_model.pkl"
TARGET = "risk_level"

# These columns are not used as prediction inputs.
# label_source is metadata. Accident-count/time columns may describe
# accidents that have already happened, which can leak the answer.
DROP_COLUMNS = ["label_source", "accidents_onsite", "accidents_time"]


def main():
    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. Available columns: {df.columns.tolist()}"
        )

    df = df.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    X = df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")
    y = df[TARGET]

    # Remove ID-like columns if present.
    id_columns = [
        col for col in X.columns
        if col.lower() == "id" or col.lower().endswith("_id")
        or col.lower().endswith("id") or "unnamed" in col.lower()
    ]
    X = X.drop(columns=id_columns, errors="ignore")

    if X.empty:
        raise ValueError("No input features remain after removing excluded columns.")

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Rows used:", len(df))
    print("Input columns:", X.columns.tolist())
    print("Risk classes:", sorted(y.unique().tolist(), key=str))
    print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

    # Save the complete pipeline and the exact input column order.
    artifact = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "target_column": TARGET,
        "risk_classes": model.named_steps["classifier"].classes_.tolist(),
        "excluded_columns": DROP_COLUMNS + id_columns,
    }
    joblib.dump(artifact, MODEL_FILE)
    print(f"\nSaved model artifact to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
