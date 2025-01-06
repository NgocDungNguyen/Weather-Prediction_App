import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import traceback
from datetime import timedelta
from flask import current_app

logger = logging.getLogger(__name__)

def process_data(filename, city, prediction_range):
    try:
        logger.info(f"Processing data for file: {filename}, city: {city}, prediction range: {prediction_range}")
        
        # Load and preprocess data
        raw_data = pd.read_csv(filename)
        raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
        
        # Select only numeric columns and datetime
        numeric_columns = ['datetime'] + raw_data.select_dtypes(include=[np.number]).columns.tolist()
        raw_data = raw_data[numeric_columns]
        
        # Remove outliers
        raw_data = remove_outliers(raw_data, 'tempmax')
        
        # Feature engineering
        raw_data = add_time_features(raw_data)
        
        # Prepare data for modeling
        feature_columns = [col for col in raw_data.columns if col not in ['datetime', 'tempmax']]
        X = raw_data[feature_columns]
        y = raw_data['tempmax']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate models
        best_model, best_model_name = train_and_evaluate_models(X_train, y_train)
        
        # Make predictions for future dates
        future_pred = predict_future(raw_data, best_model, prediction_range)
        
        # Generate and save graphs
        generate_graphs(raw_data, future_pred)
        
        logger.info("Processing completed successfully")
        
        return {
            'predictions': future_pred.to_dict(orient='records'),
            'best_model': best_model_name,
            'rmse': np.sqrt(mean_squared_error(y_test, best_model.predict(X_test))),
            'csv_filename': f"{city}_predictions.csv"
        }
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        logger.error(traceback.format_exc())
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
    return df

def train_and_evaluate_models(X_train, y_train):
    models = {
        'RandomForestReg': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearReg': LinearRegression()
    }
    
    best_model_name = None
    best_cross_val_rmse = float('inf')
    
    for name, model in models.items():
        try:
            cv_rmse_scores = -cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
            avg_rmse = np.sqrt(cv_rmse_scores.mean())
            
            if avg_rmse < best_cross_val_rmse:
                best_cross_val_rmse = avg_rmse
                best_model_name = name
        except Exception as e:
            logger.error(f"Error occurred while evaluating {name}: {str(e)}")
    
    if best_model_name is None:
        raise ValueError("No suitable model found. All models failed during evaluation.")
    
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    
    return best_model, best_model_name

def predict_future(data, model, prediction_range):
    last_date = data['datetime'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_range)
    future_data = pd.DataFrame({'datetime': future_dates})
    
    future_data = add_time_features(future_data)
    
    for col in data.columns:
        if col not in future_data.columns and col not in ['datetime', 'tempmax']:
            future_data[col] = data[col].mean()
    
    feature_columns = [col for col in future_data.columns if col not in ['datetime', 'tempmax']]
    future_pred = model.predict(future_data[feature_columns])
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
    csv_filename = "predictions.csv"
    future_data.to_csv(os.path.join(output_folder, csv_filename), index=False)