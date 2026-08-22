"""
Task 2 - Credit Card Fraud Detection
=====================================
Dataset : kartik2112/fraud-detection (Kaggle)
Models  : Logistic Regression, Decision Tree, Random Forest

Run this in a Kaggle Notebook, Google Colab, or locally.
Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn, kagglehub

    pip install kagglehub pandas numpy scikit-learn matplotlib seaborn

If running locally / on Colab, kagglehub will ask you to authenticate
with your Kaggle account the first time (or set up ~/.kaggle/kaggle.json).
On Kaggle Notebooks it works automatically.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, f1_score,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. DOWNLOAD & LOAD DATA
# ---------------------------------------------------------------------------
import kagglehub

path = kagglehub.dataset_download("kartik2112/fraud-detection")
print("Dataset downloaded to:", path)
print("Files found:", os.listdir(path))

train_path = os.path.join(path, "fraudTrain.csv")
test_path = os.path.join(path, "fraudTest.csv")

# The first column in these CSVs is an unnamed row index -> use it as index_col
df_train = pd.read_csv(train_path, index_col=0)
df_test = pd.read_csv(test_path, index_col=0)

print(f"\nTrain shape: {df_train.shape}   Test shape: {df_test.shape}")

df_train["source"] = "train"
df_test["source"] = "test"
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

print(f"Combined shape: {df.shape}")
print(f"\nOverall fraud rate: {df['is_fraud'].mean() * 100:.3f}%")
print(df["is_fraud"].value_counts().rename({0: "legit", 1: "fraud"}))

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (safe to do before the split - no target info used)
# ---------------------------------------------------------------------------
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])

# Age of the cardholder at the time of the transaction
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

# Time-based features (fraud often clusters at certain hours/days)
df["trans_hour"] = df["trans_date_trans_time"].dt.hour
df["trans_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df["trans_month"] = df["trans_date_trans_time"].dt.month


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/long points, in kilometers."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# Distance between the cardholder's home location and the merchant
df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

# Low-cardinality categoricals -> one-hot encode (no leakage risk)
df = pd.get_dummies(df, columns=["category", "gender"], drop_first=True)

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (use the dataset's own time-based split)
# ---------------------------------------------------------------------------
train_df = df[df["source"] == "train"].drop(columns=["source"]).reset_index(drop=True)
test_df = df[df["source"] == "test"].drop(columns=["source"]).reset_index(drop=True)

# High-cardinality categoricals -> frequency-encode using TRAIN stats only,
# so nothing from the test set leaks into training.
for col in ["merchant", "job", "city", "state"]:
    freq = train_df[col].value_counts(normalize=True)
    train_df[col + "_freq"] = train_df[col].map(freq)
    test_df[col + "_freq"] = test_df[col].map(freq).fillna(0.0)

drop_cols = [
    "trans_date_trans_time", "dob", "cc_num", "first", "last", "street",
    "city", "state", "zip", "job", "merchant", "trans_num", "unix_time",
    "lat", "long", "merch_lat", "merch_long",
]
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

X_train = train_df.drop(columns=["is_fraud"])
y_train = train_df["is_fraud"]
X_test = test_df.drop(columns=["is_fraud"])
y_test = test_df["is_fraud"]

print(f"\nX_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"Fraud rate -> train: {y_train.mean():.4f}   test: {y_test.mean():.4f}")

# Scale numeric features (fit on train only)
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=[np.number]).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------------------------
# Fraud is ~0.5% of transactions here, so we use class_weight="balanced"
# instead of plain accuracy-driven training, and we'll judge models on
# ROC-AUC / PR-AUC / F1 rather than accuracy.
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE
    ),
}

results = {}

for name, model in models.items():
    print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred, digits=4, target_names=["legit", "fraud"]))
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}   F1: {f1:.4f}")

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1,
    }

# ---------------------------------------------------------------------------
# 5. COMPARE MODELS
# ---------------------------------------------------------------------------
summary = pd.DataFrame({
    name: {"ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "F1": r["f1"]}
    for name, r in results.items()
}).T.sort_values("PR-AUC", ascending=False)

print("\nModel comparison (sorted by PR-AUC -- the most informative metric")
print("here since fraud is a rare class):")
print(summary)

best_name = summary.index[0]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------------------
# 6. PLOTS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")

    prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
    axes[1].plot(rec, prec, label=f"{name} (AP={r['pr_auc']:.3f})")

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("\nSaved: model_comparison.png")

# Confusion matrix for the best model
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix_best_model.png", dpi=150)
print("Saved: confusion_matrix_best_model.png")

# Feature importance (Random Forest)
if "Random Forest" in results:
    importances = (
        pd.Series(results["Random Forest"]["model"].feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance_rf.png", dpi=150)
    print("Saved: feature_importance_rf.png")

print("\nDone.")
