import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("housing.csv")

target_column = "price"

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert categorical features to numeric
X = pd.get_dummies(X)

# Train the model
model = RandomForestRegressor()
model.fit(X, y)

with open("app/reg_model.pkl", "wb") as f:
    pickle.dump(model, f)
