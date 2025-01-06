import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform, loguniform
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
import warnings
from datetime import timedelta
from flask import current_app

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a DataFrame")
warnings.filterwarnings("ignore", message="No further splits with positive gain")

# Configure KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

def process_data(filename, city, prediction_range):
    try:
        logger.info(f"Processing data for file: {filename}, city: {city}, prediction range: {prediction_range}")
        
        # Load and preprocess data
        raw_data = pd.read_csv(filename)
        raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
        raw_data.drop(columns=["name", "icon", "stations", "description"], inplace=True)
        
        # Remove outliers
        raw_data = remove_outliers(raw_data, 'tempmax')
        
        # Impute missing values
        for column in raw_data.columns:
            if raw_data[column].dtype == 'object':
                raw_data[column].fillna('Unknown', inplace=True)
            else:
                raw_data[column].fillna(raw_data[column].median(), inplace=True)
        
        # Feature engineering
        raw_data = add_time_features(raw_data)
        raw_data = add_lag_and_rolling_features(raw_data)
        
        # Prepare data for modeling
        X = raw_data.drop('tempmax', axis=1)
        y = raw_data['tempmax']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate models
        best_model, best_model_name = train_and_evaluate_models(X_train, y_train)
        
        # Fine-tune the best model
        best_model_tuned = fine_tune_model(best_model, best_model_name, X_train, y_train)
        
        # Make predictions for future dates
        future_pred = predict_future(raw_data, best_model_tuned, prediction_range)
        
        # Generate and save graphs
        generate_graphs(raw_data, future_pred)
        
        logger.info("Processing completed successfully")
        
        return {
            'predictions': future_pred.to_dict(orient='records'),
            'best_model': best_model_name,
            'rmse': np.sqrt(mean_squared_error(y_test, best_model_tuned.predict(X_test))),
            'csv_filename': f"{city}_predictions.csv"
        }
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise

def remove_outliers(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def add_time_features(df):
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    return df

def add_lag_and_rolling_features(df):
    selected_columns = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike']
    for col in selected_columns:
        for lag in [1, 2, 3]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        for window in [3, 7]:
            df[f'{col}_rolling_{window}'] = df[col].rolling(window).mean()
    return df.dropna()

def train_and_evaluate_models(X_train, y_train):
    models = {
        'RandomForestReg': RandomForestRegressor(random_state=42),
        'GradientBoostingReg': GradientBoostingRegressor(random_state=42),
        'LGBMReg': LGBMRegressor(random_state=42, verbosity=-1),
        'XGBBoost': XGBRegressor(random_state=42),
        'BayesianRidge': BayesianRidge(),
        'LinearReg': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'SVR': SVR(),
        'MLPRegressor': MLPRegressor(random_state=42),
        'PolynomialReg': make_pipeline(PolynomialFeatures(), LinearRegression())
    }
    
    best_model_name = None
    best_cross_val_rmse = float('inf')
    
    for name, model in models.items():
        cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        avg_rmse = np.sqrt(cv_rmse_scores.mean())
        
        if avg_rmse < best_cross_val_rmse:
            best_cross_val_rmse = avg_rmse
            best_model_name = name
    
    return models[best_model_name], best_model_name

def fine_tune_model(model, model_name, X_train, y_train):
    param_grids = {
        'RandomForestReg': {
            'n_estimators': randint(100, 2000),
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_features': uniform(0.1, 0.9)
        },
        'GradientBoostingReg': {
            'n_estimators': randint(100, 2000),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'subsample': uniform(0.5, 0.5)
        },
        'LGBMReg': {
            'num_leaves': randint(20, 200),
            'learning_rate': uniform(0.01, 0.2),
            'n_estimators': randint(100, 2000),
            'min_child_samples': randint(1, 50),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'verbosity': [-1]
        },
        'XGBBoost': {
            'n_estimators': randint(100, 2000),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 20),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'gamma': uniform(0, 0.5)
        },
        'BayesianRidge': {
            'alpha_1': uniform(0.001, 1),
            'alpha_2': uniform(0.001, 1),
            'lambda_1': uniform(0.001, 1),
            'lambda_2': uniform(0.001, 1)
        },
        'LinearReg': {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'positive': [True, False]
        },
        'Ridge': {
            'alpha': loguniform(1e-3, 1e2),
            'max_iter': [5000, 10000]
        },
        'Lasso': {
            'alpha': loguniform(1e-3, 1e2),
            'max_iter': [5000, 10000]
        },
        'SVR': {
            'C': loguniform(1e-2, 1e2),
            'epsilon': loguniform(1e-3, 1),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'MLPRegressor': {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (100,100), (100,50,100)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': loguniform(1e-4, 1e-1),
            'learning_rate': ['constant','adaptive'],
            'learning_rate_init': loguniform(1e-4, 1e-1),
            'max_iter': [200, 500, 1000],
            'early_stopping': [True, False],
            'momentum': uniform(0.0, 1.0),
            'nesterovs_momentum': [True, False]
        },
        'PolynomialReg': {
            'polynomialfeatures__degree': randint(2, 5),
            'linearregression__fit_intercept': [True, False]
        }
    }
    
    grid_search = RandomizedSearchCV(model, param_distributions=param_grids.get(model_name, {}),
                                     n_iter=100, cv=kfold, scoring='neg_mean_squared_error',
                                     n_jobs=-1, random_state=42, verbose=1)
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def predict_future(data, model, prediction_range):
    last_date = data['datetime'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_range)
    future_data = pd.DataFrame({'datetime': future_dates})
    
    for col in data.columns:
        if col not in ['datetime', 'tempmax']:
            future_data[col] = data[col].mean()
    
    future_data = add_time_features(future_data)
    future_data = add_lag_and_rolling_features(future_data)
    
    future_pred = model.predict(future_data.drop('datetime', axis=1))
    future_data['predicted_tempmax'] = future_pred
    
    return future_data[['datetime', 'predicted_tempmax']]

def generate_graphs(historical_data, future_data):
    output_folder = current_app.config['OUTPUT_FOLDER']
    
    # Temperature over time
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data['datetime'], historical_data['tempmax'], label='Historical')
    plt.plot(future_data['datetime'], future_data['predicted_tempmax'], label='Predicted')
    plt.title('Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'temperature_over_time.png'))
    plt.close()
    
    # Temperature distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(historical_data['tempmax'], kde=True)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.savefig(os.path.join(output_folder, 'temperature_distribution.png'))
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_data = historical_data.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'))
    plt.close()

    # Save predictions to CSV
    csv_filename = f"{historical_data['city'].iloc[0]}_predictions.csv"
    future_data.to_csv(os.path.join(output_folder, csv_filename), index=False)