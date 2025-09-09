
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# filenames
connectivity_separate_filename = 'connectivity_separate.csv'
connectivity_combined_filename = 'connectivity_combined.csv'
patient_excel_filename = 'Depression_Dataset.csv'

# NOTE: CONNECTIVITY SEPARATE needs a different approach because not all rows can be treated the same
# -> they are different regions so it is not logical to form a decision tree that treats them all with the same weight

# ---------------- patient df preprocessing ----------------

# load patient data from the excel file
patient_df = pd.read_csv(os.path.join(RAW_DIR, patient_excel_filename), usecols=['id','BDI'])
# print(sorted(patient_df['BDI'].unique()))

patient_df = patient_df.rename(columns={'id': 'patient_id'})

# all patients fit the range, but to be safe, select BDI data that fits the range (<=8 is ok, >=13 is depressed)
patient_df = patient_df[(patient_df['BDI'] <= 8) | (patient_df['BDI'] >= 13)]

# create a label
patient_df['depression'] = (patient_df['BDI'] >= 13).astype(int)

# ----------------------------------------------------------


# load connectivity data for the regions combined
connectivity_combined = pd.read_csv(os.path.join(PROCESSED_DIR, connectivity_combined_filename))

# DROP BAD PATIENTS
connectivity_combined = connectivity_combined[connectivity_combined['patient_id'] != 575]

# merge connectivity data with bdi labels
merged_df = pd.merge(connectivity_combined, patient_df[['patient_id', 'depression']], on='patient_id')

# DROP patient_id column since this isn't useful for ML
merged_df = merged_df.drop(columns=['patient_id'])

print('connectivity combined df merged successfully:')
print(merged_df.head(), '\n')
merged_df.to_csv(os.path.join(PROCESSED_DIR, 'connectivity_combined_patient.csv'), index=False)


# ---------------- model training ----------------

from sklearn.model_selection import train_test_split

# initialize features and labels
X = merged_df.drop(columns=['depression'])
y = merged_df['depression']

# split to 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# ---------------- random forest and optimization ----------------

from sklearn.ensemble import RandomForestClassifier

if input("Run grid search? [Y/N]") == 'Y':
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of each tree
        'min_samples_split': [2, 5, 10, 15, 20],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],  # Minimum number of samples required to be at a leaf node
        'max_features': ['sqrt', 'log2', None],  # Number of features to consider when looking for the best split
        'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
        'class_weight': [None, 'balanced'],  # Weighs the classes in the classification problem
    }

    # Initialize and train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # grid search setup
    grid_search = GridSearchCV(rf, param_grid, cv=5, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters: ", grid_search.best_params_)

# Best Hyperparameters:  {'bootstrap': True, 'class_weight': 'balanced', 'max_depth': None, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}

# ---------------- training ----------------

rf = RandomForestClassifier(n_estimators=200, bootstrap=True, class_weight='balanced', max_depth=None,
                            max_features=None, min_samples_leaf=4, min_samples_split=2, random_state=42)

rf.fit(X_train, y_train)

# ---------------- evaluation ----------------

from sklearn.metrics import accuracy_score, classification_report

# Predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# ---------------- feature importance ----------------

import matplotlib.pyplot as plt
import numpy as np
import textwrap

# Get feature importance
importances = rf.feature_importances_
feature_names = X.columns

# Sort in descending order
indices = np.argsort(importances)[::-1]

top_n = int(input('Plot how many of the top features?\n'))
top_indices = indices[:top_n]

# Wrap long text labels
wrapped_labels = [textwrap.fill(feature_names[i], width=20) for i in top_indices]

# Plot (Horizontal Bar Chart)
plt.figure(figsize=(10, 8))
plt.title(f"Top {top_n} Features in Depression Prediction")

plt.barh(range(top_n), importances[top_indices], align="center")
plt.yticks(range(top_n), wrapped_labels, fontsize=10)  # Decrease font size if necessary
plt.xlabel("Feature Importance Score")

plt.gca().invert_yaxis()  # Ensures the most important feature is at the top
plt.show()
