import pandas as pd
import numpy as np
from pathlib import Path
from logger import logger
import logging
from sklearn.model_selection import train_test_split
import os
import sys
import datetime
import glob
import shutil

# Adjust sys.path to include the 'project' directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, code_dir)

from src.config.config import Config
from src.config.check_structure import check_existing_file, check_existing_folder
# from logs.logger import logger

# input_filepath = Config.RAW_DATA_DIR
# output_filepath = Config.PROCESSED_DATA_DIR
def load_pre_data(input_filepath = Config.RAW_DATA_DIR):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../preprocessed).
    """
    
    logger.info('making final data set from raw data')
    logger.info(f"Data loaded from {input_filepath}.")
    

    # Prompt the user for input file paths
    # input_filepath= click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_users = os.path.join(input_filepath, "usagers-2021.csv")
    input_filepath_caract = os.path.join(input_filepath, "caracteristiques-2021.csv")
    input_filepath_places = os.path.join(input_filepath, "lieux-2021.csv")
    input_filepath_veh = os.path.join(input_filepath, "vehicules-2021.csv")
    # output_filepath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())
    
    # Call the main data processing function with the provided file paths
    return input_filepath_users, input_filepath_caract, input_filepath_places, input_filepath_veh

def combine_new_data(input_filepath_users, input_filepath_caract, input_filepath_places, input_filepath_veh, csv_files, input_filepath = Config.NEW_DATA_DIR, output_path = Config.RAW_DATA_DIR):
 
    logger.info(input_filepath)
    os.makedirs(input_filepath, exist_ok=True)

    #--Importing dataset
    df_users = pd.read_csv(input_filepath_users, sep=";", on_bad_lines='skip')
    df_caract = pd.read_csv(input_filepath_caract, sep=";", header=0, low_memory=False, on_bad_lines='skip')
    df_places = pd.read_csv(input_filepath_places, sep = ";", encoding='utf-8', on_bad_lines='skip')
    df_veh = pd.read_csv(input_filepath_veh, sep=";", on_bad_lines='skip')

    for csv_file in csv_files:
        tmp_file = csv_file
        if csv_file[0:3] == 'usa':
            tmp = pd.read_csv(tmp_file, sep=";", on_bad_lines='skip')
            df_users = pd.concat([df_users, tmp], ignore_index=True)
        elif csv_file[0:3] == 'car':
            tmp = pd.read_csv(tmp_file, sep=";", header=0, low_memory=False, on_bad_lines='skip')
            df_caract = pd.concat([df_caract, tmp], ignore_index=True)
        elif csv_file[0:3] == 'lie':
            tmp = pd.read_csv(tmp_file, sep=";", encoding='utf-8', on_bad_lines='skip')
            df_places = pd.concat([df_places, tmp], ignore_index=True)
        elif csv_file[0:3] == 'veh':
            tmp = pd.read_csv(tmp_file, sep=";", on_bad_lines='skip')
            df_veh = pd.concat([df_veh, tmp], ignore_index=True)
    
    for file, filename in zip([df_users, df_caract, df_places, df_veh], ['usagers-2021', 'caracteristiques-2021', 'lieux-2021', 'vehicules-2021']):
        output_path1 = os.path.join(output_path, f'{filename}.csv')
        file.to_csv(output_path1, index=False)


def check_new_csv(folder_path = Config.NEW_DATA_DIR):
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if csv_files:
        return csv_files
    else:
        return False

def remove_new_csv(csv_files, folder_path = Config.NEW_DATA_DIR, folder_out = Config.NEW_DATA_BACKUP_DIR):
    # Ensure destination folder exists
    os.makedirs(folder_out, exist_ok=True)
    for file in csv_files:
        shutil.move(file, folder_out)

def main():
    folder_path = Config.NEW_DATA_DIR
    csv_files = check_new_csv(folder_path)

    if csv_files:
        input_filepath = Config.RAW_DATA_DIR
        input_filepath_users, input_filepath_caract, input_filepath_places, input_filepath_veh = load_pre_data(input_filepath)

        input_filepath = Config.NEW_DATA_DIR
        output_path = Config.RAW_DATA_DIR
        combine_new_data(input_filepath_users, input_filepath_caract, input_filepath_places, input_filepath_veh, csv_files, input_filepath, output_path)

        folder_path = Config.NEW_DATA_DIR
        foler_out = Config.NEW_DATA_BACKUP_DIR
        remove_new_csv(csv_files, folder_path, foler_out)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # logger = logging.getLogger(__name__)
    main()







