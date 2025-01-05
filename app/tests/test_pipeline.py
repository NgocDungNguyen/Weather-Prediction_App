import pytest
from app.pipeline import process_data
import pandas as pd
import os

@pytest.fixture
def sample_data():
    data = {
        'datetime': pd.date_range(start='2022-01-01', end='2022-12-31'),
        'city': ['TestCity'] * 365,
        'tempmax': [20 + i % 10 for i in range(365)],
        'tempmin': [10 + i % 10 for i in range(365)],
        'temp': [15 + i % 10 for i in range(365)],
        'feelslike': [16 + i % 10 for i in range(365)],
        'humidity': [50 + i % 20 for i in range(365)],
        'wind_speed': [5 + i % 5 for i in range(365)]
    }
    df = pd.DataFrame(data)
    filename = 'test_data.csv'
    df.to_csv(filename, index=False)
    yield filename
    os.remove(filename)

def test_process_data(sample_data):
    results = process_data(sample_data, 'TestCity', 7)
    assert 'predictions' in results
    assert 'csv_filename' in results
    assert len(results['predictions']) == 7
    assert os.path.exists(os.path.join('app/outputs', results['csv_filename']))