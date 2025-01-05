from flask import Blueprint, render_template, request, jsonify, send_file
from .pipeline import process_data
from .utils import allowed_file, create_upload_folder, create_output_folder
import os

main = Blueprint('main', __name__)

UPLOAD_FOLDER = create_upload_folder()
OUTPUT_FOLDER = create_output_folder()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        city = request.form.get('city')
        prediction_range = int(request.form.get('prediction_range', 7))
        
        results = process_data(filename, city, prediction_range)
        
        return jsonify(results)
    else:
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