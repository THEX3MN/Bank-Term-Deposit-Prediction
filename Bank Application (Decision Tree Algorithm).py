# ============================================================
# BANK TERM DEPOSIT PREDICTION – DECISION TREE
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.tree import plot_tree

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
print("\n==================== DATA LOADING ====================")
data = pd.read_csv("bank dataset for assignment.csv")
print("Dataset loaded successfully")
print("Number of records:", data.shape[0])
print("Number of attributes:", data.shape[1])

# ------------------------------------------------------------
# Initial Inspection
# ------------------------------------------------------------
print("\n==================== DATA PREVIEW ====================")
print(data.head())

# ------------------------------------------------------------
# Data Cleaning
# ------------------------------------------------------------
print("\n==================== DATA CLEANING ====================")

# Check for missing values
print("Missing values per column BEFORE cleaning:")
print(data.isnull().sum())

# Remove duplicate rows
print("\nNumber of duplicate rows BEFORE cleaning:", data.duplicated().sum())
data = data.drop_duplicates()
print("Number of duplicate rows AFTER cleaning:", data.duplicated().sum())

data = data.dropna()

print("\nMissing values (NaN) per column AFTER cleaning:")
print(data.isnull().sum())

print("\nData cleaning completed.")
print("Records after cleaning:", data.shape[0])

# ------------------------------------------------------------
# Encoding of Categorical Variables
# ------------------------------------------------------------
print("\n==================== DATA ENCODING ====================")

# Target variable
data["y"] = data["y"].map({"no": 0, "yes": 1})

# Binary yes / no / unknown fields
binary_mapping = {"no": 0, "yes": 1, "unknown": -1}
data["default"] = data["default"].map(binary_mapping)
data["housing"] = data["housing"].map(binary_mapping)
data["loan"] = data["loan"].map(binary_mapping)

# Job
job_mapping = {
    "admin.": 0, "blue-collar": 1, "technician": 2, "services": 3,
    "management": 4, "retired": 5, "entrepreneur": 6,
    "self-employed": 7, "housemaid": 8, "student": 9,
    "unemployed": 10, "unknown": -1
}
data["job"] = data["job"].map(job_mapping)

# Marital status
marital_mapping = {
    "single": 0, "married": 1, "divorced": 2, "unknown": -1
}
data["marital"] = data["marital"].map(marital_mapping)

# Education level
education_mapping = {
    "illiterate": 0,
    "basic.4y": 1,
    "basic.6y": 2,
    "basic.9y": 3,
    "high.school": 4,
    "professional.course": 5,
    "university.degree": 6,
    "unknown": -1
}
data["education"] = data["education"].map(education_mapping)

# Contact type
contact_mapping = {
    "telephone": 0,
    "cellular": 1
}
data["contact"] = data["contact"].map(contact_mapping)

# Month
month_mapping = {
    "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12
}
data["month"] = data["month"].map(month_mapping)

# Day of week
day_mapping = {
    "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5
}
data["day_of_week"] = data["day_of_week"].map(day_mapping)

# Previous campaign outcome
poutcome_mapping = {
    "nonexistent": 0, "failure": 1, "success": 2
}
data["poutcome"] = data["poutcome"].map(poutcome_mapping)

# Final check
print("Remaining categorical columns:", data.select_dtypes(include=["object"]).columns)
print("Encoding completed successfully.")

# ------------------------------------------------------------
# Feature and Target Separation
# ------------------------------------------------------------
print("\n==================== FEATURE SELECTION ====================")
X = data.drop("y", axis=1)
y = data["y"]
print("Input features shape:", X.shape)
print("Target variable shape:", y.shape)

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------
print("\n==================== DATA SPLITTING ====================")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# ------------------------------------------------------------
# Model Training
# ------------------------------------------------------------
print("\n==================== MODEL TRAINING ====================")
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
print("Decision Tree model trained successfully.")

# ------------------------------------------------------------
# Model Prediction
# ------------------------------------------------------------
print("\n==================== MODEL PREDICTION ====================")
y_pred = decision_tree.predict(X_test)
print("Predictions generated for test dataset.")

# ------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------
print("\n==================== MODEL EVALUATION ====================")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(conf_matrix)

# ------------------------------------------------------------
# Visualisation: ROC Curve
# ------------------------------------------------------------
print("\n==================== ROC CURVE ====================")

y_prob = decision_tree.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_value = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label="Decision Tree")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Decision Tree")
plt.legend()
plt.tight_layout()
plt.show()

print("AUC Score:", auc_value)

# ------------------------------------------------------------
# Visualisation: Feature Importance
# ------------------------------------------------------------
print("\n==================== FEATURE IMPORTANCE ====================")
feature_importance = decision_tree.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Visualisation: Decision Tree Structure
# ------------------------------------------------------------
print("\n==================== DECISION TREE STRUCTURE ====================")

plt.figure(figsize=(22, 12))

plot_tree(
    decision_tree,
    feature_names=X.columns,
    class_names=["No Subscription", "Subscription"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)

plt.title("Decision Tree Structure (Top Levels)")
plt.tight_layout()
plt.show()
