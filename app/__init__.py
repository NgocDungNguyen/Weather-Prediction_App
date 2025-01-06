from flask import Flask
import os
import logging

def create_app():
    app = Flask(__name__, static_folder='/tmp')
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
   
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    app.config['OUTPUT_FOLDER'] = '/tmp/outputs'
    
    # Ensure the upload and output folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    from .routes import main
    app.register_blueprint(main)
    
    return app