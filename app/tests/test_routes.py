import pytest
from app import create_app
import io

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200

def test_upload_route_no_file(client):
    response = client.post('/upload')
    assert response.status_code == 400
    assert b'No file part' in response.data

def test_upload_route_empty_filename(client):
    response = client.post('/upload', data={'file': (io.BytesIO(b''), '')})
    assert response.status_code == 400
    assert b'No selected file' in response.data

def test_upload_route_invalid_file_type(client):
    response = client.post('/upload', data={'file': (io.BytesIO(b'test data'), 'test.txt')})
    assert response.status_code == 400
    assert b'Invalid file type' in response.data