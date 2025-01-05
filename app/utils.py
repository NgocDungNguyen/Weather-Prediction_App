import os

def create_upload_folder():
    upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    return upload_folder

def create_output_folder():
    output_folder = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder