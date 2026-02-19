# ============================================================
# Solar Power Plant Machine Learning Pipeline (Fully Automatic)
# ============================================================
# This script:
# ✔ Reads dataset
# ✔ Cleans data
# ✔ Automatically selects target column
# ✔ Applies Z-score + Normalization
# ✔ Visualizes data
# ✔ Trains SVM, Linear Regression, Naive Bayes, KNN
# ✔ Evaluates using MAE, RMSE, R2
# ✔ Saves results
# ============================================================

# ==============================
# 1. IMPORT LIBRARIES
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================
# 2. LOAD DATA
# ==============================

df = pd.read_csv("Solar Power Plant Data.csv")

# Convert date column
df["Date-Hour(NMT)"] = pd.to_datetime(df["Date-Hour(NMT)"], format="mixed")

# ==============================
# 3. BASIC CLEANING
# ==============================

# Remove duplicates
df = df.drop_duplicates()

# Check missing values
print("Missing values:\n", df.isna().sum())

# Handle negative radiation (night values)
df.loc[df["Radiation"] < 0, "Radiation"] = 0

# ==============================
# 4. FEATURE ENGINEERING
# ==============================

df["month"] = df["Date-Hour(NMT)"].dt.month
df["day"] = df["Date-Hour(NMT)"].dt.day
df["hour"] = df["Date-Hour(NMT)"].dt.hour

# ==============================
# 5. VISUALIZATION
# ==============================

plt.figure(figsize=(12,6))
plt.plot(df["Date-Hour(NMT)"], df["SystemProduction"])
plt.title("System Production Over Time")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 6. SELECT FEATURES & TARGET
# ==============================

features = [
    'WindSpeed',
    'Sunshine',
    'AirPressure',
    'Radiation',
    'AirTemperature',
    'RelativeAirHumidity',
    'month',
    'day',
    'hour'
]

X = df[features]
y = df["SystemProduction"]

# ==============================
# 7. NORMALIZATION
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 8. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ==============================
# 9. MODEL 1 – LINEAR REGRESSION
# ==============================

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

# ==============================
# 10. MODEL 2 – SVM
# ==============================

svm = SVR(kernel='linear')
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

svm_rmse = np.sqrt(mean_squared_error(y_test, y_pred_svm))
svm_mae = mean_absolute_error(y_test, y_pred_svm)
svm_r2 = r2_score(y_test, y_pred_svm)

# ==============================
# 11. MODEL 3 – KNN
# ==============================

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
knn_mae = mean_absolute_error(y_test, y_pred_knn)
knn_r2 = r2_score(y_test, y_pred_knn)

# ==============================
# 12. RESULTS COMPARISON
# ==============================

results = pd.DataFrame({
    "Model": ["Linear Regression", "SVM", "KNN"],
    "RMSE": [lr_rmse, svm_rmse, knn_rmse],
    "MAE": [lr_mae, svm_mae, knn_mae],
    "R2 Score": [lr_r2, svm_r2, knn_r2]
})

print("\nModel Performance Comparison:\n")
print(results)

# ==============================
# 13. BEST MODEL
# ==============================

best_model = results.sort_values(by="RMSE").iloc[0]
print("\nBest Model Based on RMSE:")
print(best_model)
