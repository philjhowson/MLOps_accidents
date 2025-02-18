import pandas as pd
import numpy as np
from pathlib import Path
from logger import logger
import logging
from sklearn.model_selection import train_test_split
import os
import sys
import datetime

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
itemNoAll = 100
def main(input_filepath = Config.RAW_DATA_DIR, output_filepath = Config.NEW_DATA_DIR):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info(f"Data loaded from {input_filepath}.")
    logger.info(output_filepath)

    os.makedirs(output_filepath, exist_ok=True)

    # Prompt the user for input file paths
    # input_filepath= click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_users = os.path.join(input_filepath, "usagers-2021.csv")
    input_filepath_caract = os.path.join(input_filepath, "caracteristiques-2021.csv")
    input_filepath_places = os.path.join(input_filepath, "lieux-2021.csv")
    input_filepath_veh = os.path.join(input_filepath, "vehicules-2021.csv")
    # output_filepath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())
    
    # Call the main data processing function with the provided file paths
    process_data(input_filepath_users, input_filepath_caract, input_filepath_places, input_filepath_veh, output_filepath, itemNoAll)

def process_data(input_filepath_users, input_filepath_caract, input_filepath_places, input_filepath_veh, output_folderpath, itemNoAll):
 
    #--Importing dataset
    df_users = pd.read_csv(input_filepath_users, sep=";")
    df_caract = pd.read_csv(input_filepath_caract, sep=";", header=0, low_memory=False)
    df_places = pd.read_csv(input_filepath_places, sep = ";", encoding='utf-8')
    df_veh = pd.read_csv(input_filepath_veh, sep=";")

    # print(df_users.head(), len(df_users), df_caract.head(), len(df_caract), df_places.head(), len(df_places), df_veh.head(), len(df_veh))
    # randomly select 100 items
    allChoices = df_users['Num_Acc'].unique()

    selected_items = np.random.choice(allChoices, size=itemNoAll, replace=False)
    # print(selected_items)

    df_users_1 = df_users[df_users['Num_Acc'].isin(selected_items)]
    df_caract_1 = df_caract[df_caract['Num_Acc'].isin(selected_items)]
    df_places_1 = df_places[df_places['Num_Acc'].isin(selected_items)]
    df_veh_1 = df_veh[df_veh['Num_Acc'].isin(selected_items)]

    for file, filename in zip([df_users_1, df_caract_1, df_places_1, df_veh_1], ['usagers', 'caracteristiques', 'lieux', 'vehicules']):
        current_date = datetime.datetime.now().strftime('%Y%m%d')
        output_filepath = os.path.join(output_folderpath, f'{filename}-{current_date}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
