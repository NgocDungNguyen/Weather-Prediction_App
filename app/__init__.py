from flask import Flask
import os
import logging

def create_app():
    app = Flask(__name__)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
   
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    app.config['OUTPUT_FOLDER'] = '/tmp/outputs'
    app.static_folder = '/tmp'  # This allows serving files from /tmp
    
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    from .routes import main
    app.register_blueprint(main)
    
    return app