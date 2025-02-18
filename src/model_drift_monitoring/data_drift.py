# import libraries
import json
import pandas as pd
import numpy as np
from sklearn import datasets
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, RegressionQualityMetric, RegressionErrorPlot, RegressionErrorDistribution, DataDriftTable
from evidently.metrics import (
    ClassificationQualityMetric,
    ConfusionMatrixMetric,
    RocCurveMetric,
    DatasetDriftMetric,
    DataDriftTable,
    TargetDriftMetric,
)
from evidently.preset import ClassificationPreset
from logger import logger
import os

# Import the necessary modules
from evidently.pipeline.column_mapping import ColumnMapping


from src.config.check_structure import check_existing_file, check_existing_folder
from src.config.config import Config


import pandas as pd
import numpy as np
import zipfile
from sklearn import ensemble
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.ui.workspace import Workspace
from evidently.test_suite import TestSuite
from evidently.test_preset import DataQualityTestPreset
from evidently.tests import TestColumnValueMean
from evidently.metrics import (DatasetSummaryMetric, 
                               DatasetCorrelationsMetric,
                               DatasetMissingValuesMetric)
import logging

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

    
    # Run the report
    data_drift_dataset_report.run(reference_data=data_ref, 
                                    current_data=data_cur,
                                    column_mapping=column_mapping)

    

    # Convert the JSON string to a Python dictionary for pretty printing
    report_data = json.loads(data_drift_dataset_report.json())

    return report_data, data_drift_dataset_report, data_ref, data_cur

def create_and_run_test_suite(ref_data, curr_data, SAVE_FILE = False):
    """
    Create and run a test suite for dataset summary.
    """
    # Define tests for dataset summary
    tests = [TestColumnValueMean(column_name=col) for col in ref_data.columns if ref_data[col].dtype != 'object']

    test_suite = TestSuite(tests=[DataQualityTestPreset()] + tests)
    # test_suite = TestSuite(tests=[DataQualityTestPreset(),
    #                               TestColumnValueMean(column_name=ref_data.columns)])
    # Run the test suite
    test_suite.run(reference_data=ref_data, current_data=curr_data)

    # save the report as HTML
    if SAVE_FILE:
        test_suite.save_html(f"{Config.DATA_DRIFT_MONOTOR_DIR}templates/data_drift_suite.html")
    return test_suite

def create_and_run_report(reference_data, current_data):
    """
    Generates a classification report using Evidently.
    """
    # Create a Report instance for classification with a set of predefined metrics
    # TODO : Add the previously imported metrics

    # Initialize the report with desired metrics
    classification_report = Report(metrics=[
        ClassificationQualityMetric(),
        ConfusionMatrixMetric(),
        RocCurveMetric(),
        DatasetDriftMetric(),
        DataDriftTable(),
        TargetDriftMetric()
        ])


    # report = Report(metrics=[
    #     DatasetSummaryMetric(), 
    #     DatasetCorrelationsMetric(),
    #     DatasetMissingValuesMetric(),
    # ])

    # TODO : Generate the report using reference_data and current_data
    report.run(reference_data=reference_data, current_data=current_data)

    return report

def add_report_and_test_suite_to_workspace(workspace, project_name, 
                                           project_description, 
                                           report, test_suite):
    """
    Adds a report and test suite to an existing or new project in a workspace.
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    for item in [report, test_suite]:
        workspace.add_report(project.id, item)
        print(f"New item added to project {project_name}")


def main():
    """
    Main function to train the model and save it.
    """
    logger.info("starting checking data drifting between training and testing data.")

    # Define constants for workspace and project details
    WORKSPACE_NAME = f"{Config.DATA_DRIFT_MONOTOR_DIR}accidents-workspace"
    PROJECT_NAME = "data_monitoring_test_suite_and_report"
    PROJECT_DESCRIPTION = "Evidently Dashboards"


    # Path to the joblib file containing train and test datasets
    file_path = Config.PROCESSED_DATA_DIR#'data/preprocessed/'

    # Base filename for the output model file
    output_file_path = Config.DATA_DRIFT_MONOTOR_DIR 
    check_existing_folder(output_file_path)

    report_data, data_drift_dataset_report, ref_data, curr_data = loadData(file_path)

    # Save the report in JSON format with indentation for better readability
    save_filename = f"{output_file_path}data_drift_report.json"
    print(save_filename)
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, 'w') as f:
        json.dump(report_data, f, indent=4)

    # save HTML
    save_filename = f"{output_file_path}templates/Classification Report.html"
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    data_drift_dataset_report.save_html(save_filename)

    logging.info('Creating and running test suite...')
    test_suite = create_and_run_test_suite(ref_data, curr_data)
    logging.info('Test suite completed.')

    logging.info('Creating and running report...')
    report = create_and_run_report(ref_data, curr_data)
    logging.info('Report generated successfully.')

    logging.info('Adding report to workspace...')
    workspace = Workspace(WORKSPACE_NAME)
    add_report_and_test_suite_to_workspace(workspace, PROJECT_NAME, 
                                           PROJECT_DESCRIPTION, test_suite, report)
    logging.info('Report added to workspace.')
    # in a notebook run :
    # data_drift_dataset_report.show()
    
    logger.info("data drifting check completed.")
    logger.info("-----------------------------------")

if __name__ == "__main__":
    main()