import os

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def create_upload_folder():
    upload_folder = os.path.join(os.getcwd(), 'app', 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    return upload_folder

def create_output_folder():
    output_folder = os.path.join(os.getcwd(), 'app', 'outputs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder