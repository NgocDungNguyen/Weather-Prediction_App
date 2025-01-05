import os

def create_upload_folder():
    upload_folder = '/tmp/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    return upload_folder

def create_output_folder():
    output_folder = '/tmp/outputs'
    os.makedirs(output_folder, exist_ok=True)
    return output_folder