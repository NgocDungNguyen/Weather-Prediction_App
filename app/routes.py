import os
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from .pipeline import process_data
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/upload', methods=['POST'])
def upload():
    current_app.logger.info("Upload route hit")
    if 'file' not in request.files:
        current_app.logger.error("No file part")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        current_app.logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        current_app.logger.info(f"Saving file to: {filepath}")
        try:
            file.save(filepath)
        except Exception as e:
            current_app.logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': f"Error saving file: {str(e)}"}), 500
        
        city = request.form.get('city')
        prediction_range = int(request.form.get('prediction_range', 7))
        
        current_app.logger.info(f"Processing data for city: {city}, prediction range: {prediction_range}")
        
        try:
            results = process_data(filepath, city, prediction_range)
            current_app.logger.info("Data processed successfully")
            return jsonify(results)
        except Exception as e:
            current_app.logger.error(f"Error processing data: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        current_app.logger.error("Invalid file type")
        return jsonify({'error': 'Invalid file type'}), 400

@main.route('/process', methods=['POST'])
def process():
    data = request.json
    filename = data.get('filename')
    city = data.get('city')
    prediction_range = data.get('prediction_range', 7)
    
    results = process_data(filename, city, prediction_range)
    
    return jsonify(results)

@main.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)