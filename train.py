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