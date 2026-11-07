"""
prepare_data.py - shared data preparation for the HVAC fault classifier.

Both teammates import this file so they train on exactly the same data.

    from prepare_data import load_split
    X_train, y_train, X_test, y_test, meta_train, meta_test = load_split()

Run it directly to check the sizes:  python src/prepare_data.py
"""
import sys

import pandas as pd

# ---------------------------------------------------------------- settings --
SEED = 42                      # use this random seed in every model
DATA_FILE = "data/may_combined.csv"   # run scripts from the repo's main folder

# Time-block split (same dates for every scenario). Last week of May is test;
# the day before it is dropped as a gap so neighbouring minutes don't leak.
TEST_START = pd.Timestamp("2018-05-25 00:00")   # test = May 25-31
GAP_START = pd.Timestamp("2018-05-24 00:00")    # May 24 dropped as a gap

ROLL_WINDOWS = [15, 60]        # minutes

# The 5 signals from the proposal, plus supply-air temperature (where the
# coi_bias fault shows up most), plus the occupancy flag.
# SA_TEMPSPT is constant (55.18 F) so it is not a feature, only used in sa_resid.
BASE_SIGNALS = ["OA_TEMP", "MA_TEMP", "RA_TEMP", "OA_DMPR_DM", "OA_DMPR", "SA_TEMP"]
OCCUPANCY = "SYS_CTL"          # 1 = occupied, 0 = unoccupied

LABEL_MAP = {                  # fault_type in the combined file -> class label
    "normal": "fault_free",
    "damper_stuck": "damper_stuck",
    "coi_bias": "sensor_bias",    
}


# ------------------------------------------------------------------ loading --
def load_raw(path=DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])

    df["label"] = df["fault_type"].map(LABEL_MAP)
    df["scenario"] = df["fault_type"] + "_" + df["fault_level"].astype(str)
    return df.sort_values(["scenario", "Datetime"]).reset_index(drop=True)


# ----------------------------------------------------------------- features --
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Damper command vs. actual position (large when the damper is stuck).
    df["damper_resid"] = df["OA_DMPR"] - df["OA_DMPR_DM"]

    # Measured mixed-air temp vs. what the damper position predicts.
    # (Approximation: damper position treated as the outdoor-air fraction.)
    expected_ma = df["RA_TEMP"] + df["OA_DMPR"] * (df["OA_TEMP"] - df["RA_TEMP"])
    df["ma_resid"] = df["MA_TEMP"] - expected_ma

    # Supply-air temp vs. its setpoint (shifted when the SA sensor is biased).
    df["sa_resid"] = df["SA_TEMP"] - df["SA_TEMPSPT"]

    # Rolling mean / std, computed inside each scenario so windows never
    # cross from one scenario into the next.
    roll_cols = ["damper_resid", "ma_resid", "sa_resid", "MA_TEMP", "OA_DMPR"]
    g = df.groupby("scenario", sort=False)
    for w in ROLL_WINDOWS:
        for c in roll_cols:
            r = g[c].rolling(w, min_periods=w)
            df[f"{c}_mean{w}"] = r.mean().reset_index(level=0, drop=True)
            df[f"{c}_std{w}"] = r.std().reset_index(level=0, drop=True)

    # Drop the first rows of each scenario where the windows aren't full.
    return df.dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list:
    base = BASE_SIGNALS + [OCCUPANCY, "damper_resid", "ma_resid", "sa_resid"]
    rolled = [c for c in df.columns if "_mean" in c or "_std" in c]
    return base + rolled


# -------------------------------------------------------------------- split --
def load_split(path=DATA_FILE):
    df = add_features(load_raw(path))
    cols = feature_columns(df)

    train = df[df["Datetime"] < GAP_START]
    test = df[df["Datetime"] >= TEST_START]
    meta = ["Datetime", "scenario", "fault_level"]   # for per-scenario reports only

    return (train[cols], train["label"], test[cols], test["label"],
            train[meta], test[meta])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE
    X_tr, y_tr, X_te, y_te, m_tr, m_te = load_split(path)
    print(f"Features ({X_tr.shape[1]}): {list(X_tr.columns)}\n")
    print(f"Train: {X_tr.shape[0]} rows  ({m_tr.Datetime.min()} -> {m_tr.Datetime.max()})")
    print(f"Test:  {X_te.shape[0]} rows  ({m_te.Datetime.min()} -> {m_te.Datetime.max()})\n")
    print("Train classes:\n", y_tr.value_counts().to_string(), "\n")
    print("Test classes:\n", y_te.value_counts().to_string(), "\n")
    print("Rows per scenario (train / test):")
    print(pd.concat([m_tr.scenario.value_counts().rename("train"),
                     m_te.scenario.value_counts().rename("test")], axis=1).to_string())