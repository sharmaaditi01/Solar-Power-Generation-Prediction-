#  Solar Power Generation Forecasting using Machine Learning

This project predicts solar power output using meteorological time-series data.

##  Problem Statement
Accurate prediction of solar energy production helps improve grid stability,
energy planning, and renewable integration.

##  Data Features
- Wind Speed
- Sunshine Duration
- Air Pressure
- Solar Radiation
- Air Temperature
- Relative Humidity
- Time-based features (hour, day, month)

##  Models Implemented
- Linear Regression
- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (applied after discretizing output into Low/Medium/High production levels)

Naive Bayes was used as a comparative probabilistic approach by converting the continuous target variable into categorical bins.

##  Best Model Performance (KNN)
- **R² Score:** 0.78
- **RMSE:** 658
- **MAE:** 284
- **Normalized RMSE:** 9.1%

##  Tech Stack
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn

## Results
Model	RMSE	MAE	R² Score	Normalized RMSE
Linear Regression	885	479	0.61	12.2%
SVR	894	443	0.60	12.3%
KNN	658	284	0.78	9.1%

KNN Regressor achieved the best performance, showing the ability to capture nonlinear environmental relationships affecting solar output.


## Evaluation Metrics
RMSE (Root Mean Squared Error): Measures magnitude of prediction error.
MAE (Mean Absolute Error): Average absolute deviation from actual values.
R² Score: Explains how well the model captures variance in solar production.
Normalized RMSE: Scale-independent metric for fair comparison.
A Normalized RMSE of ~9% indicates strong predictive performance for real-world energy forecasting data.

## Key Insights

Solar production has nonlinear dependence on radiation and time-of-day.
Linear models underperform due to environmental variability.
KNN effectively captures localized production patterns.
Temporal feature engineering significantly improved prediction quality.


## How to Run

Download the Repository
Download ZIP or clone:

git clone https://github.com/yourusername/solar-power-forecasting-ml.git
cd solar-power-forecasting-ml

## Install Required Libraries
pip install -r requirements.txt

## Add Dataset
Place the dataset file inside the project folder:

Solar Power Plant Data.csv
The dataset is already included in this repository:
Solar Power Plant Data.csv
No additional download is required.


## Run the Script
python solar_forecasting.py

## Output
The script will:
Clean and preprocess the dataset
Perform exploratory visualization
Train multiple ML models
Compare performance using RMSE, MAE, and R²
Display the best-performing model

## Tech Stack
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Machine Learning

## Future Improvements
Implement ensemble models like Random Forest / XGBoost
Apply Time-Series Cross Validation
Integrate real-time weather forecast data
Deploy as a web-based solar prediction dashboard

## Project Type
Academic Machine Learning Project focused on Renewable Energy Forecasting, demonstrating an end-to-end ML workflow from preprocessing to model evaluation.
Install dependencies:

