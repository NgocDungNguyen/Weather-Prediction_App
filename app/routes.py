from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from .pipeline import process_data
from werkzeug.utils import secure_filename
import os
import traceback
import logging

main = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload():
    logger.info("Upload route hit")
    try:
        if 'file' not in request.files:
            logger.error("No file part")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            logger.info(f"Saving file to: {filepath}")
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file.save(filepath)
            
            city = request.form.get('city', 'NewYork, NY, US')
            prediction_range = int(request.form.get('prediction_range', 7))
            
            logger.info(f"Form data - City: {city}, Prediction Range: {prediction_range}")
            logger.info(f"Processing data for city: {city}, prediction range: {prediction_range}")
            
            results = process_data(filepath, city, prediction_range)
            logger.info("Data processed successfully")
            return jsonify(results)
        else:
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Error in upload route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@main.route('/download/<filename>')
def download(filename):
    try:
        return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)
    except Exception as e:
        logger.error(f"Error in download route: {str(e)}")
        return jsonify({'error': 'File not found'}), 404