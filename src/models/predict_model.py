import joblib
import pandas as pd
import sys
import json
import glob

from logger import logger
from src.config.config import Config

# Load your saved model
# loaded_model = joblib.load("./src/models/trained_model.joblib")
def load_model(model_path):
    """
    Load the machine learning model from the specified path.

    Parameters:
    model_path (str): The file path to the saved model.

    Returns:
    model: The loaded model.
    """
    model = load(model_path)
    return model

def make_predictions(model, new_data):
    """
    Make predictions using the provided model and new data.

    Parameters:
    model: The trained model used for making predictions.
    new_data (DataFrame or array-like): The data for which predictions are to be made.

    Returns:
    array: The predictions made by the model.
    """
    predictions = model.predict(new_data)
    return predictions

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

def main():
    """
    Main function to load datasets, load the model, make predictions, and save the predictions.
    """
    logger.info("(5) Starting the model prediction process.")

    # Check if a new model version signal file exists
    new_model_signal_file = f'{Config.TRAINED_MODEL_DIR}signal_new_model_version'
    if not os.path.exists(new_model_signal_file):
        logger.info("No new model version found. Skipping inference.")
        logger.info("-----------------------------------")
        return

    try:
        # Path to the joblib file containing train and test datasets
        file_path = Config.PROCESSED_DATA_DIR

        # Load the train and test datasets
        # Load train and test datasets from joblib file
        X_test = pd.read_csv(f'{file_path}X_test.csv')
        y_test = pd.read_csv(f'{file_path}y_test.csv')
        y_test = np.ravel(y_test)

        # Path to the base filename of the model
        base_model_filename = Config.TRAINED_MODEL_DIR

        # Find the latest versioned model file
        latest_model_file = find_latest_versioned_model(base_model_filename)
        
        print('Last version model path:')
        print(latest_model_file)

        # Load the model
        model = load(latest_model_file)
        
        # Make predictions using the model
        predictions = make_predictions(model, X_test)
        print(f"Predictions: {predictions}")
        
        # Save predictions to a CSV file
        output_file_path = Config.OUTPUT_PREDICTIONS_RESULTS_FILE
        pd.DataFrame(predictions, columns=['Prediction']).to_csv(output_file_path, index=False)
        logger.info("Prediction file data saved successfully.")
        logger.info(output_file_path)

        # Each service script creates its signal file at the end
        open('signal_inference_done', 'w').close()

        logger.info("Model inference completed.")
        logger.info("-----------------------------------")
    
    finally:
        # Remove the signal_new_model_version file regardless of success or failure
        if os.path.exists(new_model_signal_file):
            os.remove(new_model_signal_file)
            logger.info("Removed signal_new_model_version file.")
            print("Removed signal_new_model_version file.")

if __name__ == "__main__":
    main()
