from flask import Blueprint, render_template, request, jsonify, send_file
from .pipeline import process_data
from werkzeug.utils import secure_filename
import os

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        city = request.form.get('city')
        prediction_range = int(request.form.get('prediction_range', 7))
        
        try:
            results = process_data(filepath, city, prediction_range)
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
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