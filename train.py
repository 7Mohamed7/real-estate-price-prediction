import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset and clean column names
my_data = pd.read_csv('Real_estate.csv')
my_data.columns = my_data.columns.str.strip()

# 2. Separate features (X) and target (y)
X = my_data.drop(columns=['No', 'price'])
y = my_data['price']

# 3. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Print dataset splitting details
print("=" * 45)
print("          DATASET SPLITTING DETAILS          ")
print("=" * 45)
print(f"Total Dataset Samples : {len(my_data)}")
print(
    f"Training Set (X_train): {X_train.shape[0]} samples"
    f" ({X_train.shape[0]/len(my_data):.1%})"
)
print(
    f"Testing Set  (X_test) : {X_test.shape[0]} samples"
    f" ({X_test.shape[0]/len(my_data):.1%})"
)
print(f"Features Count        : {X_train.shape[1]}")
print("=" * 45 + "\n")

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training
my_model = LinearRegression()
my_model.fit(X_train_scaled, y_train)

# 6. Model Evaluation
y_pred = my_model.predict(X_test_scaled)

train_accuracy = my_model.score(X_train_scaled, y_train) * 100
test_accuracy = my_model.score(X_test_scaled, y_test) * 100

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("=" * 45)
print("         MODEL PERFORMANCE & ACCURACY        ")
print("=" * 45)
print(f"Training Accuracy (R2 Score) : {train_accuracy:.2f}%")
print(f"Testing Accuracy  (R2 Score) : {test_accuracy:.2f}%")
print("-" * 45)
print(f"Mean Absolute Error (MAE)    : {mae:.2f}")
print(f"Mean Squared Error (MSE)     : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("=" * 45 + "\n")

# 7. Save Model and Scaler using joblib
joblib.dump(my_model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("[INFO] Model and Scaler successfully saved as .pkl files!")
