# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Your exact feature set (lower-case)
FEATURE_COLS_DEFAULT = [
    "alcohol",
    "density",
    "volatile acidity",
    "total sulfur dioxide",
    "chlorides",
    "free sulfur dioxide",
    "sulphates",
    "ph",                 # <= note lower-case
    "residual sugar",
    "citric acid",
    "fixed acidity",
]

TARGET_COL_DEFAULT = "quality"

def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize headers once (lower-case)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def get_xy(df: pd.DataFrame, feature_cols=None, target_col: str = TARGET_COL_DEFAULT):
    # drop non-chemical meta columns if present
    for col in ["type", "id", "name"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if feature_cols is None:
        feature_cols = FEATURE_COLS_DEFAULT

    # select only the desired features
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    # classifiers want integer labels
    if y.dtype.kind not in "iu":
        y = y.astype(int)
    return X, y

def split_train_valid(X, Y, test_size=0.2, random_state=42, stratify=True):
    strat = Y if stratify else None
    return train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=strat)
