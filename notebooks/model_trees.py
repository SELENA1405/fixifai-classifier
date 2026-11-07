# %% [markdown]
# # Person A - Decision Tree and Random Forest
# Data: May 2018 (data/may_combined.csv) - fault-free, damper stuck, sensor bias (coi_bias).
# Run this from the repo's main folder. It uses src/prepare_data.py so that
# both teammates train on exactly the same train/test data.

# %% Step 1: Import libraries
import os
import sys
sys.path.append("src")            # lets Python find src/prepare_data.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, accuracy_score)

from prepare_data import load_split, SEED

os.makedirs("results", exist_ok=True)

# %% Step 2: Load the shared train / test data
X_train, y_train, X_test, y_test, info_train, info_test = load_split()

class_names = ["fault_free", "damper_stuck", "sensor_bias"]

print("X_train:", X_train.shape, " X_test:", X_test.shape)

# list that will collect one row of results per model
all_results = []

# %% Step 3: Decision Tree (small max_depth so it stays readable)
tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=SEED)
tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

# %% Step 4: Evaluate the Decision Tree
print(classification_report(y_test, tree_pred, labels=class_names, zero_division=0))

tree_cm = confusion_matrix(y_test, tree_pred, labels=class_names)
print(pd.DataFrame(tree_cm, index=class_names, columns=class_names))

ConfusionMatrixDisplay(tree_cm, display_labels=class_names).plot()
plt.title("Decision Tree - confusion matrix")
plt.savefig("results/confusion_decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Step 5: Draw the tree (easy to explain to technicians)
plt.figure(figsize=(22, 9))
plot_tree(tree, feature_names=list(X_train.columns), class_names=tree.classes_,
          filled=True, fontsize=8)
plt.savefig("results/decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Step 6: Save the Decision Tree results
report = classification_report(y_test, tree_pred, labels=class_names, output_dict=True, zero_division=0)
row = {"model": "Decision Tree (max_depth=4)",
       "macro_f1": f1_score(y_test, tree_pred, average="macro"),
       "accuracy": accuracy_score(y_test, tree_pred)}
for c in class_names:
    row[f"precision_{c}"] = report[c]["precision"]
    row[f"recall_{c}"] = report[c]["recall"]
all_results.append(row)

# %% Step 7: Tune the Random Forest with grouped cross-validation
# Groups = blocks of ~5 days. Each fold holds out whole blocks of time, so
# minutes next to each other never end up in both train and validation.
# (We group by time, not by scenario: fault_free and sensor_bias have only one
#  scenario each, so leaving a scenario out would remove a whole class.)
groups = info_train["Datetime"].dt.day // 5   # May 1-4, 5-9, 10-14, 15-19, 20-23

# every 5th minute only, to keep tuning fast
X_small = X_train.iloc[::5]
y_small = y_train.iloc[::5]
groups_small = groups.iloc[::5]

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, 20],
    "min_samples_leaf": [1, 5],
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
    param_grid,
    scoring="f1_macro",
    cv=GroupKFold(n_splits=5),
)
grid.fit(X_small, y_small, groups=groups_small)

print("Best settings:", grid.best_params_)
print("Best CV macro-F1:", round(grid.best_score_, 3))

# %% Step 8: Train the final Random Forest on ALL training data
rf = RandomForestClassifier(**grid.best_params_, class_weight="balanced",
                            random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# %% Step 9: Evaluate the Random Forest
print(classification_report(y_test, rf_pred, labels=class_names, zero_division=0))

rf_cm = confusion_matrix(y_test, rf_pred, labels=class_names)
print(pd.DataFrame(rf_cm, index=class_names, columns=class_names))

ConfusionMatrixDisplay(rf_cm, display_labels=class_names).plot()
plt.title("Random Forest - confusion matrix")
plt.savefig("results/confusion_random_forest.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Step 10: Which features matter most?
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)
print(importances.head(10))

importances.to_csv("results/feature_importance_random_forest.csv", header=["importance"])

# %% Step 11: Accuracy for each scenario (which fault levels are hard?)
scenario_check = info_test.copy()
scenario_check["true"] = y_test.values
scenario_check["pred"] = rf_pred
scenario_check["correct"] = scenario_check["true"] == scenario_check["pred"]

per_scenario = scenario_check.groupby("scenario")["correct"].mean().round(3)
print(per_scenario)

per_scenario.to_csv("results/per_scenario_accuracy_random_forest.csv", header=["accuracy"])

# %% Step 12: Save the Random Forest results and the summary file
report = classification_report(y_test, rf_pred, labels=class_names, output_dict=True, zero_division=0)
row = {"model": f"Random Forest {grid.best_params_}",
       "macro_f1": f1_score(y_test, rf_pred, average="macro"),
       "accuracy": accuracy_score(y_test, rf_pred)}
for c in class_names:
    row[f"precision_{c}"] = report[c]["precision"]
    row[f"recall_{c}"] = report[c]["recall"]
all_results.append(row)

results_table = pd.DataFrame(all_results).round(3)
print(results_table.T)

# Person A writes their own file so it never clashes with Person B's file.
results_table.to_csv("results/results_person_a.csv", index=False)