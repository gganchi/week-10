# your code here

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import pickle
from sklearn.tree import DecisionTreeRegressor


# Loading the coffee analysis data from the CSV file
URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
data = pd.read_csv(URL)
"""
Exercise 1.
Training the scikit-learn linear regression model to predict the 
'rating' based on the '100g_USD' feature.
"""
# Using a consistent name
df = data

# Linear regression training details
df_lr = df[["rating", "100g_USD"]].dropna()
X_lr = df_lr[["100g_USD"]].values
y = df_lr["rating"].values

lr = LinearRegression()
lr.fit(X_lr, y)

# Save the trained model as a pickle file called model_1.pickle
joblib.dump(lr, "model_1.pickle")
print("Saved: model_1.pickle (LinearRegression on 100g_USD → rating)")

"""
Exercise 2.
Training the Decision Tree Regression model to make prediction.
"""
# Converting the roast column into numerical labels
roast_map = {cat: idx for idx, cat in enumerate(data["roast"].unique())}
data["roast_num"] = data["roast"].map(roast_map)

# Train DecisionTreeRegressor on 100g_USD and roast_num
feature_cols = ["100g_USD", "roast_num"]
train = data[["rating"] + feature_cols].dropna(subset=["rating", "100g_USD", "roast_num"])
X = train[feature_cols].values
y = train["rating"].values

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X, y)

joblib.dump(dtr, "model_2.pickle")
print("Saved: model_2.pickle (DecisionTreeRegressor on 100g_USD & roast_num → rating)")
