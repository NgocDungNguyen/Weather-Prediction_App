from flask import Flask
import os

def create_app():
    app = Flask(__name__)
       
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
       
    from .routes import main
    app.register_blueprint(main)
       
    return app