from sklearn.model_selection import train_test_split
from src.data.check_structure import check_existing_file, check_existing_folder
import os
import pandas as pd
import logging
def split_data(df, output_folderpath):
    target = df['grav']
    feats = df.drop(['grav'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state = 42)

    #--Filling NaN values
    col_to_fill_na = ["surf", "circ", "col", "motor"]
    X_train[col_to_fill_na] = X_train[col_to_fill_na].fillna(X_train[col_to_fill_na].mode().iloc[0])
    X_test[col_to_fill_na] = X_test[col_to_fill_na].fillna(X_train[col_to_fill_na].mode().iloc[0])

    # drop id_usager from train and test set
    # X_train.drop(['id_usager'], axis=1, inplace=True)
    # X_test.drop(['id_usager'], axis=1, inplace=True)

    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)
    

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('split data set for trainig')
    df = pd.read_csv('./data/preprocessed/training_data.csv', index_col = False)
    split_data(df, './data/preprocessed/')