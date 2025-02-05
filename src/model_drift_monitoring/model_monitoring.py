import numpy as np
import pandas as pd
import joblib
import glob
import os
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.ui.workspace import Workspace
from evidently.pipeline.column_mapping import ColumnMapping

from src.config.check_structure import check_existing_file, check_existing_folder
from src.config.config import Config

def find_latest_versioned_model(base_filename):
    """
    Find the latest versioned model file based on base_filename.
    Returns the path to the latest versioned model file.
    """
    search_pattern = f"{base_filename}-v*-*.joblib"
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No model files found with pattern '{search_pattern}'")
    
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def get_prediction(file_path, base_model_filename):
    """
    Generates predictions for reference and current data using RandomForestClassifier.
    """
    # Create a copy of the dataframes to avoid modifying the original data
    # Load train and test datasets
    X_train = pd.read_csv(f'{file_path}X_train.csv')
    X_test = pd.read_csv(f'{file_path}X_test.csv')
    y_train = pd.read_csv(f'{file_path}y_train.csv')
    y_test = pd.read_csv(f'{file_path}y_test.csv')
    reference_data = X_train.copy()
    reference_data['target'] = y_train
    current_data = X_test.copy()
    current_data['target'] = y_test

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Create a simple RandomForestClassifier
    # TODO : fit the model on the reference dataset
    # Note that the target name is 'target'

    # Path to the base filename of the model
    # base_model_filename = Config.TRAINED_MODEL_DIR
    # Find the latest versioned model file
    latest_model_file = find_latest_versioned_model(base_model_filename)
    print('Last version model path:')
    print(latest_model_file)
    # Load the model
    model = joblib.load(latest_model_file)

    # Generate predictions for reference and current data
    reference_data['prediction'] = model.predict_proba(X_train)[:, 1]
    current_data['prediction'] = model.predict_proba(X_test)[:, 1]

    return reference_data, current_data

def generate_classification_report(reference_data, current_data):
    """
    Generates a classification report using Evidently.
    """
    # TODO : Create a Report instance for classification with a set of predefined metrics.
    # Use the ClassificationPreset with probas_threshold=0.5
    classification_report = Report(metrics=[
        ClassificationPreset(probas_threshold=0.5),
    ])

    # Generate the report
    classification_report.run(reference_data=reference_data, current_data=current_data)

    return classification_report

def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
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
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")

if __name__ == "__main__":
    # Defining workspace and project details
    output_file_path = Config.DATA_DRIFT_MONOTOR_DIR
    WORKSPACE_NAME = f"{output_file_path}accidents-workspace"
    PROJECT_NAME = "rf_model_monitoring"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Get predictions for reference and current data
    # TODO : complete
    file_path = Config.PROCESSED_DATA_DIR
    base_model_filename = Config.OUTPUT_TRAINED_MODEL_FILE_RF 
    reference_data, current_data = get_prediction(file_path, base_model_filename)

    # Generate the classification report
    # TODO : complete
    classification_report = generate_classification_report(reference_data, current_data)

    # Set workspace
    workspace = Workspace(WORKSPACE_NAME)

    # Add report to workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, classification_report)