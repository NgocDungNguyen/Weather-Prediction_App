from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
    
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    from .routes import main
    app.register_blueprint(main)
    
    return app