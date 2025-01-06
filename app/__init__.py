from flask import Flask
import os
import logging

def create_app():
    app = Flask(__name__, static_folder='outputs', static_url_path='/static')
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
   
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
    app.config['OUTPUT_FOLDER'] = os.path.join(app.root_path, 'outputs')
    
    # Ensure the upload and output folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    from .routes import main
    app.register_blueprint(main)
    
    return app
