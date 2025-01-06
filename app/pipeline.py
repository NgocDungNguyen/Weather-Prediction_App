import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import logging
from datetime import timedelta
from flask import current_app

logger = logging.getLogger(__name__)

def process_data(filename, city, prediction_range):
    try:
        logger.info(f"Processing data for file: {filename}, city: {city}, prediction range: {prediction_range}")
        
        # Load data
        raw_data = pd.read_csv(filename, parse_dates=['datetime'])
        logger.info(f"Data loaded. Shape: {raw_data.shape}")
        
        # Add time features
        raw_data['day_of_year'] = raw_data['datetime'].dt.dayofyear
        raw_data['month'] = raw_data['datetime'].dt.month
        
        # Select features
        features = ['temp', 'humidity', 'windspeed', 'day_of_year', 'month']
        X = raw_data[features]
        y = raw_data['tempmax']
        logger.info(f"Features selected: {features}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Data split completed")
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Make predictions
        future_pred = predict_future(raw_data, model, prediction_range, features)
        logger.info("Future predictions made")
        
        # Generate graphs
        graph_paths = generate_graphs(raw_data, future_pred)
        logger.info("Graphs generated")
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        
        return {
            'predictions': future_pred.to_dict(orient='records'),
            'best_model': 'RandomForestRegressor',
            'rmse': rmse,
            'csv_filename': f"{city}_predictions.csv",
            'graph_paths': graph_paths
        }
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise

def predict_future(data, model, prediction_range, features):
    last_date = data['datetime'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_range)
    future_data = pd.DataFrame({'datetime': future_dates})
    
    future_data['day_of_year'] = future_data['datetime'].dt.dayofyear
    future_data['month'] = future_data['datetime'].dt.month
    
    for feature in features:
        if feature not in future_data.columns:
            future_data[feature] = data[feature].mean()
    
    future_pred = model.predict(future_data[features])
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
    temp_over_time_path = os.path.join(output_folder, 'temperature_over_time.png')
    plt.savefig(temp_over_time_path)
    plt.close()
    
    # Save predictions to CSV
    csv_filename = "predictions.csv"
    csv_path = os.path.join(output_folder, csv_filename)
    future_data.to_csv(csv_path, index=False)

    return {
        'temperature_over_time': '/static/outputs/temperature_over_time.png',
        'csv_file': f'/static/outputs/{csv_filename}'
    }