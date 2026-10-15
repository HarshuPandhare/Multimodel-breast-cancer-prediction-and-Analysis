import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = "data/METABRIC_RNA_Mutation.csv"

os.makedirs("models", exist_ok=True)



# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# FEATURES
# =========================================================

features = [
    "er_status",
    "pr_status",
    "her2_status",
    "age_at_diagnosis",
    "tumor_size",
    "lymph_nodes_examined_positive"
]

target = "pam50_+_claudin-low_subtype"


# =========================================================
# CHECK COLUMNS
# =========================================================

for col in features + [target]:

    if col not in df.columns:

        raise ValueError(
            f"Column not found: {col}"
        )


# =========================================================
# SELECT DATA
# =========================================================

data = df[features + [target]].copy()


# =========================================================
# ENCODE ER / PR / HER2
# =========================================================

def encode_marker(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["positive", "pos", "1", "true", "yes"]:
        return 1

    if value in ["negative", "neg", "0", "false", "no"]:
        return 0

    try:

        value = float(value)

        if value == 1:
            return 1

        if value == 0:
            return 0

    except:
        pass

    return np.nan


for col in [
    "er_status",
    "pr_status",
    "her2_status"
]:

    data[col] = data[col].apply(
        encode_marker
    )


# =========================================================
# NUMERIC FEATURES
# =========================================================

numeric_features = [
    "age_at_diagnosis",
    "tumor_size",
    "lymph_nodes_examined_positive"
]

for col in numeric_features:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


# =========================================================
# REMOVE MISSING VALUES
# =========================================================

data = data.dropna()

print("Clean dataset:", data.shape)


# =========================================================
# TARGET ENCODING
# =========================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    data[target].astype(str)
)

X = data[features]


print("\nClasses:")

for i, name in enumerate(
    target_encoder.classes_
):

    print(i, "=", name)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# MODELS
# =========================================================

models = {

    "Random Forest": RandomForestClassifier(

        n_estimators=500,

        max_features="sqrt",

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    ),


    "Extra Trees": ExtraTreesClassifier(

        n_estimators=500,

        max_features="sqrt",

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    ),


    "XGBoost": XGBClassifier(

        n_estimators=500,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="multi:softprob",

        eval_metric="mlogloss",

        random_state=42,

        n_jobs=-1
    ),


    "LightGBM": LGBMClassifier(

        n_estimators=500,

        learning_rate=0.05,

        max_depth=-1,

        num_leaves=31,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        verbosity=-1,

        n_jobs=-1
    ),


    "HistGradientBoosting": HistGradientBoostingClassifier(

        max_iter=300,

        learning_rate=0.05,

        max_leaf_nodes=31,

        random_state=42
    ),


    "Logistic Regression": LogisticRegression(

        max_iter=3000,

        class_weight="balanced",

        random_state=42
    )
}


# =========================================================
# TRAIN AND COMPARE
# =========================================================

results = []

trained_models = {}


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    results.append({

        "Model": name,

        "Accuracy": accuracy

    })

    trained_models[name] = model

    print(
        f"{name}: {accuracy * 100:.2f}%"
    )


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)


print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# SELECT BEST MODEL
# =========================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_accuracy = results_df.iloc[0]["Accuracy"]


print("\n")
print("=" * 60)
print("BEST TEST RESULT")
print("=" * 60)

print(
    "Model:",
    best_model_name
)

print(
    "Test Accuracy:",
    f"{best_accuracy * 100:.2f}%"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

best_prediction = best_model.predict(
    X_test
)

print("\nClassification Report:\n")

print(
    classification_report(

        y_test,

        best_prediction,

        target_names=target_encoder.classes_,

        zero_division=0

    )
)


# =========================================================
# SAVE BEST MODEL
# =========================================================

model_package = {

    "model": best_model,

    "target_encoder": target_encoder,

    "features": features,

    "accuracy": best_accuracy,

    "model_name": best_model_name,

    "classes": target_encoder.classes_.tolist()

}


joblib.dump(

    model_package,

    "models/breast_cancer_model.pkl"

)


print("\n")
print("=" * 60)

print(
    "BEST MODEL SAVED"
)

print(
    "models/breast_cancer_model.pkl"
)

print("=" * 60)