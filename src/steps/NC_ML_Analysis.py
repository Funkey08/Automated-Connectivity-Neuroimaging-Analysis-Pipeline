

import os
import numpy as np
import pandas as pd
import re
import glob

data_folder = "C:/Users/fmava/Downloads/EEG Learning/MDD Dataset Practice/Files"

connectivity_files = glob.glob(os.path.join(data_folder, "*CACOH"))

# Extract summary statistics
def extract_features(filepath):
    data = np.load(filepath)
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "max": np.max(data),
        "min": np.min(data)
    }


# --------------------- Prepare Connectivity Data ---------------------
# Create a DataFrame with the features

features_list = []

for file in connectivity_files:
    filename = os.path.basename(file)
    patient_id = filename.split("_")[0]
    # extract region name
    region = '_'.join(filename.split("_")[3:-1])

    features = extract_features(file)
    features["patient_id"] = patient_id
    features["region"] = region
    features_list.append(features)

features_df = pd.DataFrame(features_list)
# SAVE DATAFRAME FOR TIME
features_df.to_pickle("features_df.pkl")
# LOAD DATAFRAME
# features_df = pd.read_pickle("features_df.pkl")

# Pivot so each region now becomes a column
features_df = features_df.pivot(index="patient_id", columns="region").reset_index()
features_df.columns = ['_'.join(col).strip() for col in features_df.columns.values]  # Flatten MultiIndex


# --------------------- LOAD PATIENT LABELS (BDI Scores) ---------------------

patient_info = pd.read_excel("Data_4_Import_REST.xlsx")

patient_info["depressed"] = patient_info["BDI"].apply(lambda x: 1 if x > 13 else (0 if x <= 8 else np.nan))

# Drop rows where BDI is ambiguous (8 < BDI <= 13)
patient_info = dropna(subset=["depressed"])

# Merge features with patient labels
# TODO: figure out how to merge even if some patient files / rows / IDs are missing
df = pd.merge(features_df, patient_info, left_on = "patient_id_", right_on = "patient_id").drop(columns=["patient_id"])

# --------------------- PREPROCESS DATA ---------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = df.dropna()
X = df.drop(columns=["depressed"])
y = df["depressed"]

# Standardize the features
# TODO: is this step necessary?
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SAVE X_SCALED
np.save("X_scaled.npy", X_scaled)
# LOAD X_SCALED IF NEEDED
# X_scaled = np.load("X_scaled.npy")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# -------------------- ML TRAINING --------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------- VISUALIZATION --------------------

import matplotlib.pyplot as plt

feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind="barh")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")
plt.show()
