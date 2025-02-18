from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn import ensemble

import mlflow
import mlflow.sklearn

import sklearn
import pandas as pd 
import joblib
import numpy as np
from logger import logger
import os
import datetime
import json
# import dagshub

from src.config.check_structure import check_existing_file, check_existing_folder
from src.config.config import Config

print(joblib.__version__)

def train_model(file_path, output_base_filename, log_to_mlflow=True):
    """
    Train a RandomForestClassifier model on the provided dataset and log metrics with MLFlow.

    Parameters:
    file_path (str): Path to the joblib file containing the train and test datasets.
    output_base_filename (str): Base filename for saving the model.
    log_to_mlflow (bool): Whether to log metrics to MLFlow.

    Returns:
    model: Trained RandomForestClassifier model.
    """

    # Load train and test datasets
    X_train = pd.read_csv(f'{file_path}X_train.csv')
    X_test = pd.read_csv(f'{file_path}X_test.csv')
    y_train = pd.read_csv(f'{file_path}y_train.csv')
    y_test = pd.read_csv(f'{file_path}y_test.csv')
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Number of trees in random forest
    n_estimators = [200] #[int(x) for x in np.linspace(start = 200, stop = 250, num = 2)] #(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = [10] #[int(x) for x in np.linspace(1, 10, num = 2)] # ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [30] #[int(x) for x in np.linspace(10, 30, num = 2)] # [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5]#[2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4]#[1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]#[True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    rf = ensemble.RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)
    # Fit the random search model
    model = rf_random.fit(X_train, y_train)

    # Print the selected parameters for reference
    print(f"Selected parameters: {model.best_estimator_}")
    
    # Predict the target variable for the test set
    probabilities = model.predict_proba(X_test)[:, 1] 
    predictions = (probabilities > 0.5).astype(int)

    # Calculate accuracy of the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    
    # Generate versioned filename
    version = 1
    while True:
        versioned_filename = generate_versioned_filename(output_base_filename, version)
        if not os.path.exists(versioned_filename):
            break
        version += 1

    
    # if log_to_mlflow:
    #     # When we run it via github action, it gives an error: dagshub does not have init function.
    #     # This is because github action's dagshub version is different.
    #     # For github actions we use user-token env variables.
    #     # dagshub.init("nba_mlops", "joelaftreth", mlflow=True)
    #     dagshub.init("MLOps_accidents", "philjhowson", mlflow=True)

    

    if log_to_mlflow:
        # Extract just the filename without the path and extension for the run name
        run_name = os.path.splitext(os.path.basename(versioned_filename))[0]

        # # Initialize MLFlow tracking
        # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080/#/"))  # MLFlow tracking server URI
        # # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/philjhowson/MLOps_accidents.mlflow"))
        # mlflow.set_experiment("accidents_prediction")  # Experiment name

        # Define tracking_uri
        client = mlflow.MlflowClient(tracking_uri="http://localhost:8080")

        # Define experiment name, run name and artifact_path name
        accident_experiment = mlflow.set_experiment("accident_prediction") 

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("best_parameters", model.best_estimator_)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)

            # Log the trained model
            mlflow.sklearn.log_model(model, "model")

    return model, accuracy, versioned_filename


def save_metrics(metrics_file_path, metrics):
    """
    Save the given metrics to the specified file.
    
    Parameters:
    metrics_file_path (str): Path to the file where the metrics will be saved.
    metrics (dict): Dictionary containing the metrics to save.
    """
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f)


def load_best_metrics(metrics_file_path):
    """
    Load the best metrics from the specified file.
    
    Parameters:
    metrics_file_path (str): Path to the file containing the best metrics.
    
    Returns:
    dict: Dictionary containing the best metrics.
    """
    if not os.path.exists(metrics_file_path):
        save_metrics(metrics_file_path, {"accuracy": 0})

    with open(metrics_file_path, 'r') as f:
        tmp = json.load(f)
        return tmp
    # if os.path.exists(metrics_file_path):
    #     with open(metrics_file_path, 'r') as f:
    #         return json.load(f)
    # else:
    #     return {'accuracy': 0}

def generate_versioned_filename(base_filename, version):
    """
    Generate a versioned filename based on base_filename, version number, and current date.
    Example: If base_filename='model', version=1, it will generate 'model-v1-20240628.joblib'
    """
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    return f"{base_filename}-v{version}-{current_date}.joblib"


def main():
    """
    Main function to train the model and save it.
    """
    logger.info("(4) Starting the model training process.")

    # Path to the joblib file containing train and test datasets
    file_path = Config.PROCESSED_DATA_DIR#'data/preprocessed/'

    # Base filename for the output model file
    base_output_file_path = Config.OUTPUT_TRAINED_MODEL_FILE_RF #'models/'

    # Base filename for the discarded model file
    discarded_output_file_path = Config.OUTPUT_TRAINED_MODEL_FILE_RF_DISCARDED #'models/'


    # Train the random forest classifier model
    log_to_mlflow = os.getenv("LOG_TO_MLFLOW", "true").lower() == "true"
    model, new_accuracy, versioned_filename = train_model(file_path, base_output_file_path, log_to_mlflow)
    print(f"New Accuracy: {new_accuracy}")

    metrics_file = f'{Config.TRAINED_MODEL_DIR}best_model_metrics.json'
    
    # Load the best accuracy from metrics file
    best_metrics = load_best_metrics(metrics_file)
    best_accuracy = best_metrics.get('accuracy', 0)

    # Print new accuracy
    print(f"New Accuracy: {new_accuracy}")
    print(f"Best current Accuracy: {best_accuracy}")

    if new_accuracy > best_accuracy:
        # Save the model to the original path
        check_existing_folder(versioned_filename)
        joblib.dump(model, versioned_filename)
        logger.info("Model file data saved successfully.")
        logger.info(versioned_filename)

        # Save new metrics as the best metrics
        new_metrics = {'accuracy': new_accuracy}
        save_metrics(metrics_file, new_metrics)

        # Create the signal file indicating a new model version
        open(f'{Config.TRAINED_MODEL_DIR}signal_new_model_version', 'w').close()
        print("Creating signal file models/signal_new_model_version...")
    else:
        # Save the model to the discarded path
        discarded_filename = generate_versioned_filename(discarded_output_file_path, 1)
        check_existing_file(discarded_filename)
        joblib.dump(model, discarded_filename)
        logger.info("Model file data saved in discarded folder.")
        logger.info(discarded_filename)
    
    # Always create this signal file at the end of model training
    open('signal_model_training_done', 'w').close()
    
    logger.info("Model training completed.")
    logger.info("-----------------------------------")

if __name__ == "__main__":
    main()