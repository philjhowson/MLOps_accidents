# import libraries
import json
import pandas as pd
import numpy as np
from sklearn import datasets
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric
from logger import logger
import os

# Import the necessary modules
from evidently.pipeline.column_mapping import ColumnMapping


from src.config.check_structure import check_existing_file, check_existing_folder
from src.config.config import Config

def loadData(file_path):
    """
    load data .

    Parameters:
    file_path (str): Path to the joblib file containing the train and test datasets.
    
    Returns:
    model: Trained RandomForestClassifier model.
    """

    # Load train and test datasets
    X_train = pd.read_csv(f'{file_path}X_train.csv')
    X_test = pd.read_csv(f'{file_path}X_test.csv')
    y_train = pd.read_csv(f'{file_path}y_train.csv')
    y_test = pd.read_csv(f'{file_path}y_test.csv')
    # y_train = np.ravel(y_train)
    # y_test = np.ravel(y_test)

    X_train['target'] = y_train
    X_test['target'] = y_test


    # Create a new instance of the ColumnMapping class
    column_mapping = ColumnMapping()

    # Set the target column name
    column_mapping.target = 'target'

    # Set the prediction column name
    column_mapping.prediction = None

    # Set the list of numerical feature column names
    numerical_features = ['secu1','victim_age','catv','obsm','motor','circ','surf','situ','vma','atm','col','lat','long','nb_victim','nb_vehicules','hour']
    column_mapping.numerical_features = numerical_features

    # Set the list of categorical feature column names
    categorical_features = ['place', 'catu', 'sexe', 'year_acc', 'catr','jour','mois','lum','dep','com','agg_','int_']
    column_mapping.categorical_features = categorical_features

    # Split the dataset for drift detection into reference and current data
    data_ref = X_train
    data_cur = X_test

    # Initialize the report with desired metrics
    data_drift_dataset_report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),    
    ])

    # Run the report
    data_drift_dataset_report.run(reference_data=data_ref, 
                                    current_data=data_cur,
                                    column_mapping=column_mapping)

    

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data = json.loads(data_drift_dataset_report.json())

    return report_data, data_drift_dataset_report



def main():
    """
    Main function to train the model and save it.
    """
    logger.info("starting checking data drifting between training and testing data.")

    # Path to the joblib file containing train and test datasets
    file_path = Config.PROCESSED_DATA_DIR#'data/preprocessed/'

    # Base filename for the output model file
    output_file_path = Config.DATA_DRIFT_MONOTOR_DIR 
    check_existing_folder(output_file_path)

    report_data, data_drift_dataset_report = loadData(file_path)

    # Save the report in JSON format with indentation for better readability
    save_filename = f"{output_file_path}data_drift_report.json"
    print(save_filename)
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, 'w') as f:
        json.dump(report_data, f, indent=4)

    # save HTML
    data_drift_dataset_report.save_html(f"{output_file_path}Classification Report.html")

    # in a notebook run :
    # data_drift_dataset_report.show()
    
    logger.info("data drifting check completed.")
    logger.info("-----------------------------------")

if __name__ == "__main__":
    main()