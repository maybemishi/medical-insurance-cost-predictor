import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

medical_cost = pd.read_csv("medical_insurance.csv")


medical_cost.replace({'sex': {'female': 0, 'male': 1}}, inplace=True)
medical_cost.replace({'smoker': {'no': 0, 'yes': 1}}, inplace=True)
print("Encoded 'sex' and 'smoker' columns.")


medical_cost = pd.get_dummies(medical_cost, columns=['region'], drop_first=True, dtype=int)
print("Created new columns for regions.")

X = medical_cost.drop(columns=['charges'])
y = medical_cost['charges']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
print("Model training complete.")

# --- Model Evaluation ---
# 1. Make predictions on the test set
y_pred = model.predict(X_test)

# 2. Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 3. Convert error metrics to INR
usd_to_inr_rate = 1.87  # Approximate conversion rate
mae_inr = mae * usd_to_inr_rate
rmse_inr = rmse * usd_to_inr_rate

print("\n--- Model Performance ---")
print(f"Random Forest Regressor R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): ₹{mae_inr:,.2f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse_inr:,.2f}")

import pickle
pickle_out=open("model.pkl","wb")
pickle.dump(model,pickle_out)
pickle_out.close()

