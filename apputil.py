# your code here

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib

URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

def main():
    # --- Load data ---
    df = pd.read_csv(URL)
    """
    Exercise 1.
    Creating the train.py file.
    """
    # Linear Regression: rating ~ 100g_USD
    df_lr = df[["rating", "100g_USD"]].dropna()
    X_lr = df_lr[["100g_USD"]].values
    y = df_lr["rating"].values

    lr = LinearRegression()
    lr.fit(X_lr, y)

    joblib.dump(lr, "model_1.pickle")
    print("Saved: model_1.pickle (LinearRegression on 100g_USD → rating)")

    """
    Exercise 2.
    Training the Decision Tree Regression model to make prediction.
    """
    # Build roast_cat from roast (categorical → numeric)
    if "roast" not in df.columns:
        # Create an empty column if missing so code still runs
        df["roast"] = np.nan

    df["roast_cat"] = df["roast"].apply(roast_category)

    # Train DecisionTreeRegressor on 100g_USD and roast_cat
    feature_cols = ["100g_USD", "roast_cat"]
    df_dt = df[["rating"] + feature_cols].dropna(subset=["rating", "100g_USD", "roast_cat"])
    X_dt = df_dt[feature_cols].values
    y_dt = df_dt["rating"].values

    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_dt, y_dt)

    joblib.dump(dtr, "model_2.pickle")
    print("Saved: model_2.pickle (DecisionTreeRegressor on 100g_USD & roast_cat → rating)")
