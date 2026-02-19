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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================
# 1. Read Dataset
# =====================

FILE_PATH = "Solar Power Plant Data.csv"
df = pd.read_csv(FILE_PATH)

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)

# =====================
# 2. Data Cleaning
# =====================

print("\nCleaning Dataset...")

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("Cleaning Completed")
print("New Shape:", df.shape)

# =====================
# 3. Automatic Target Selection
# =====================
# Assumes LAST numeric column is the output (standard ML format)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    raise ValueError("Dataset must contain multiple numeric columns.")

TARGET_COLUMN = numeric_cols[-1]   # Automatically chosen
FEATURE_COLUMNS = numeric_cols[:-1]

print(f"\nAutomatically Selected Target Column: {TARGET_COLUMN}")

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# =====================
# 4. Preprocessing
# Z-score Standardization + Normalization
# =====================

print("\nApplying Scaling (Z-score + MinMax)...")

scaling_pipeline = Pipeline([
    ("zscore", StandardScaler()),
    ("normalize", MinMaxScaler())
])

X_scaled = scaling_pipeline.fit_transform(X)

# =====================
# 5. Train-Test Split
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Train-Test Split Completed")

# =====================
# 6. Visualization
# =====================

print("\nGenerating Visualizations...")

plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

df[numeric_cols].hist(figsize=(12,10), bins=25)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("feature_distributions.png")
plt.close()

print("Plots saved as PNG files")

# =====================
# 7. Initialize Models
# =====================

models = {
    "Support Vector Machine": SVR(kernel='rbf'),
    "Linear Regression": LinearRegression(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
}

# =====================
# 8. Evaluation Function
# =====================

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name} Results")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return mae, rmse, r2

# =====================
# 9. Train & Compare Models
# =====================

results = {}

for name, model in models.items():
    results[name] = evaluate_model(name, model)

# =====================
# 10. Save Results
# =====================

results_df = pd.DataFrame(results, index=["MAE", "RMSE", "R2"]).T
results_df.to_csv("model_comparison_results.csv")

print("\nModel comparison saved to model_comparison_results.csv")

# =====================
# 11. Save Processed Dataset
# =====================

processed_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
processed_df[TARGET_COLUMN] = y.values
processed_df.to_csv("processed_dataset.csv", index=False)

print("Processed dataset saved.")

print("\nPipeline Execution Completed Successfully.")
# ============================================================
