import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/METABRIC_RNA_Mutation.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "breast_cancer_model.pkl"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

features = [
    "er_status",
    "pr_status",
    "her2_status",
    "age_at_diagnosis",
    "tumor_size",
    "lymph_nodes_examined_positive"
]

target = "pam50_+_claudin-low_subtype"


# ============================================================
# CHECK COLUMNS
# ============================================================

print("\nChecking columns...")

missing_columns = []

for column in features + [target]:

    if column not in df.columns:
        missing_columns.append(column)


if missing_columns:

    print("\nERROR: The following columns were not found:")

    for column in missing_columns:
        print(" -", column)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise SystemExit


print("All required columns found.")


# ============================================================
# SELECT DATA
# ============================================================

data = df[features + [target]].copy()


# ============================================================
# DISPLAY ORIGINAL VALUES
# ============================================================

print("\nOriginal ER values:")
print(data["er_status"].value_counts(dropna=False))

print("\nOriginal PR values:")
print(data["pr_status"].value_counts(dropna=False))

print("\nOriginal HER2 values:")
print(data["her2_status"].value_counts(dropna=False))


# ============================================================
# NORMALIZE BIOMARKER VALUES
# ============================================================

def encode_marker(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    positive_values = [
        "positive",
        "pos",
        "1",
        "true",
        "yes"
    ]

    negative_values = [
        "negative",
        "neg",
        "0",
        "false",
        "no"
    ]

    if value in positive_values:
        return 1

    if value in negative_values:
        return 0

    # Try numerical values
    try:

        numeric_value = float(value)

        if numeric_value == 1:
            return 1

        if numeric_value == 0:
            return 0

    except:
        pass

    return np.nan


for column in [
    "er_status",
    "pr_status",
    "her2_status"
]:

    data[column] = data[column].apply(
        encode_marker
    )


# ============================================================
# CONVERT NUMERIC FEATURES
# ============================================================

numeric_columns = [
    "age_at_diagnosis",
    "tumor_size",
    "lymph_nodes_examined_positive"
]

for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

before = len(data)

data = data.dropna()

after = len(data)

print("\nRows before cleaning:", before)
print("Rows after cleaning:", after)
print("Rows removed:", before - after)


# ============================================================
# CHECK TARGET
# ============================================================

print("\nTarget classes:")

print(
    data[target].value_counts()
)


# ============================================================
# ENCODE TARGET
# ============================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    data[target].astype(str)
)


# ============================================================
# INPUT FEATURES
# ============================================================

X = data[features].copy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

# Check whether stratified split is possible
class_counts = pd.Series(y).value_counts()

if class_counts.min() >= 2:

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

else:

    print(
        "Warning: Some classes have fewer than 2 samples."
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1,

    max_features="sqrt"
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n====================================")
print("MODEL RESULTS")
print("====================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(
            len(target_encoder.classes_)
        ),
        target_names=target_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(
        len(target_encoder.classes_)
    )
)

print(cm)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})


importance = importance.sort_values(

    by="Importance",

    ascending=False

)


print("\nFeature Importance:")

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {

    "model": model,

    "target_encoder": target_encoder,

    "features": features,

    "accuracy": accuracy,

    "classes": target_encoder.classes_.tolist()

}


joblib.dump(

    model_package,

    MODEL_PATH

)


print("\n====================================")
print("MODEL SAVED")
print("====================================")

print(
    "Saved to:",
    MODEL_PATH
)

print("\nClasses:")

for i, class_name in enumerate(
    target_encoder.classes_
):

    print(
        i,
        "->",
        class_name
    )


print("\nTraining finished successfully.")