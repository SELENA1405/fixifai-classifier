# %% [markdown]
# # HVAC fault classification - data preparation
# Goal: predict the fault class (fault_free / damper_stuck / sensor_bias)
# from sensor readings. Each row is one minute of readings.

# %% Step 1: Import libraries
import pandas as pd

# %% Step 2: Load the data
df = pd.read_csv("data/new_datasets_april_combined.csv", parse_dates=["Datetime"])

print(df.shape)
df.head()

# %% Step 3: Look at the scenarios we have
print(df.groupby(["fault_type", "fault_level"]).size())

# %% Step 4: Keep only ONE sensor-bias scenario
# The four oa_bias files were identical, so we keep +4 and drop the rest.
# (Mention this as a limitation in the report.)
df = df[(df["fault_type"] != "oa_bias") | (df["fault_level"] == 4.0)]
df = df.reset_index(drop=True)

print(df.groupby(["fault_type", "fault_level"]).size())

# %% Step 5: Create the target (y) column called "label"
label_names = {
    "normal": "fault_free",
    "damper_stuck": "damper_stuck",
    "oa_bias": "sensor_bias",
}
df["label"] = df["fault_type"].map(label_names)

# a name for each scenario (used later for reports and rolling features)
df["scenario"] = df["fault_type"] + "_" + df["fault_level"].astype(str)

# sort so that the rows of each scenario are in time order
df = df.sort_values(["scenario", "Datetime"]).reset_index(drop=True)

print(df["label"].value_counts())

# %% Step 6: Feature engineering - two difference features
# 1) damper position minus damper command (big gap when the damper is stuck)
df["damper_resid"] = df["OA_DMPR"] - df["OA_DMPR_DM"]

# 2) measured mixed-air temp minus the temp we would expect from the damper
#    position (approximation: damper position = outdoor-air fraction)
expected_ma = df["RA_TEMP"] + df["OA_DMPR"] * (df["OA_TEMP"] - df["RA_TEMP"])
df["ma_resid"] = df["MA_TEMP"] - expected_ma

# %% Step 7: Feature engineering - rolling mean and std over the last 15 and 60 minutes
# We do this inside each scenario so a window never mixes two scenarios.
cols_to_roll = ["damper_resid", "ma_resid", "MA_TEMP", "OA_DMPR"]

for window in [15, 60]:
    for col in cols_to_roll:
        by_scenario = df.groupby("scenario")[col]
        df[f"{col}_mean{window}"] = by_scenario.transform(lambda s: s.rolling(window).mean())
        df[f"{col}_std{window}"] = by_scenario.transform(lambda s: s.rolling(window).std())

# %% Step 8: Remove rows with missing values
# The first rows of every scenario have no full window yet, so they are NaN.
print("rows before:", len(df))
df = df.dropna().reset_index(drop=True)
print("rows after: ", len(df))

# %% Step 9: Choose the features (X) and the target (y)
feature_cols = (
    ["OA_TEMP", "MA_TEMP", "RA_TEMP", "OA_DMPR_DM", "OA_DMPR", "SYS_CTL",
     "damper_resid", "ma_resid"]
    + [c for c in df.columns if "_mean" in c or "_std" in c]
)

X = df[feature_cols]
y = df["label"]

print(len(feature_cols), "features")
print(feature_cols)

# %% Step 10: Train / test split BY TIME (not random)
# Rows next to each other (minute by minute) are almost identical, so a random
# split would put near-copies of the training rows in the test set and give
# unrealistically good scores. Instead:
#   train = April 1 - 22, gap = April 23 (dropped), test = April 24 - 30
train_mask = df["Datetime"] < "2018-04-23"
test_mask = df["Datetime"] >= "2018-04-24"

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# keep date / scenario / fault level for reports (NOT for training)
info_train = df.loc[train_mask, ["Datetime", "scenario", "fault_level"]]
info_test = df.loc[test_mask, ["Datetime", "scenario", "fault_level"]]

# %% Step 11: Check the sizes (both teammates should get the same numbers)
print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("\nTrain classes:\n", y_train.value_counts())
print("\nTest classes:\n", y_test.value_counts())

RANDOM_STATE = 42   # use this in every model, e.g. RandomForestClassifier(random_state=42)