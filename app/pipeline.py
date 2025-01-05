import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
import traceback
from datetime import timedelta

logger = logging.getLogger(__name__)

OUTPUT_FOLDER = '/tmp/outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_data(filename, city, prediction_range):
    try:
        logger.info(f"Processing data for file: {filename}, city: {city}, prediction range: {prediction_range}")
        
        # Load and preprocess data
        df = pd.read_csv(filename)
        logger.info(f"Data loaded. Shape: {df.shape}")
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        logger.info(f"Data filtered for city {city}. Shape: {df.shape}")
        
        # Feature engineering
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Prepare features and target
        features = ['day_of_year', 'month', 'day_of_week', 'is_weekend', 'humidity', 'wind_speed', 'tempmin', 'temp', 'feelslike']
        target = 'tempmax'
        
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define preprocessing steps
        numeric_features = ['day_of_year', 'month', 'humidity', 'wind_speed', 'tempmin', 'temp', 'feelslike']
        categorical_features = ['day_of_week', 'is_weekend']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train the model
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        
        model.fit(X_train, y_train)
        
        # Make predictions
        last_date = df['datetime'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_range)
        future_features = pd.DataFrame({
            'day_of_year': future_dates.dayofyear,
            'month': future_dates.month,
            'day_of_week': future_dates.dayofweek,
            'is_weekend': future_dates.dayofweek.isin([5, 6]).astype(int),
            'humidity': [df['humidity'].mean()] * prediction_range,
            'wind_speed': [df['wind_speed'].mean()] * prediction_range,
            'tempmin': [df['tempmin'].mean()] * prediction_range,
            'temp': [df['temp'].mean()] * prediction_range,
            'feelslike': [df['feelslike'].mean()] * prediction_range
        })
        
        predictions = model.predict(future_features)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': future_dates,
            'predicted_tempmax': predictions
        })
        
        # Generate graphs
        generate_graphs(df, results_df)
        
        # Save results to CSV
        csv_filename = f"{city}_predictions.csv"
        results_df.to_csv(os.path.join(OUTPUT_FOLDER, csv_filename), index=False)
        
        logger.info(f"Processing completed. Results saved to {csv_filename}")
        
        return {
            'predictions': results_df.to_dict(orient='records'),
            'csv_filename': csv_filename
        }
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_graphs(historical_data, predictions):
    try:
        logger.info("Generating graphs")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Temperature over time
        plt.figure(figsize=(12, 6))
        plt.plot(historical_data['datetime'], historical_data['tempmax'], label='Historical')
        plt.plot(predictions['date'], predictions['predicted_tempmax'], label='Predicted')
        plt.title('Temperature Over Time')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'temperature_over_time.png'))
        plt.close()
        
        # Temperature distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(historical_data['tempmax'], kde=True)
        plt.title('Temperature Distribution')
        plt.xlabel('Temperature')
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'temperature_distribution.png'))
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_features = ['tempmax', 'humidity', 'wind_speed', 'tempmin', 'temp', 'feelslike']
        sns.heatmap(historical_data[corr_features].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'correlation_heatmap.png'))
        plt.close()
        
        logger.info("Graphs generated successfully")
    except Exception as e:
        logger.error(f"Error in generate_graphs: {str(e)}")
        logger.error(traceback.format_exc())
        raise