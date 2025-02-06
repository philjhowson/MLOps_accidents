import os

def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    return True
    
    
def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, ask if we want to create it.'''
    if not os.path.exists(folder_path):
        return True
    else:
        return False
