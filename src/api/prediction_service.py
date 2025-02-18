from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from joblib import load
import glob
# from src.config.config import Config

# Construct the path to the 'trained_models' directory
# Get the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Construct the path to the 'trained_models' directory
trained_models_dir = os.path.join(project_root, 'models')

# Specify the base filename of the trained model
base_joblib_filename = 'model_best_rf'


def find_latest_versioned_model(base_filename):
    """
    Find the latest versioned model file based on base_filename.
    Returns the path to the latest versioned model file.
    """
    search_pattern = f"{base_filename}-v*-*.joblib"
    files = glob.glob(os.path.join(trained_models_dir, search_pattern))

    if not files:
        raise FileNotFoundError(f"No model files found with pattern '{search_pattern}'")

    latest_file = max(files, key=os.path.getctime)
    return latest_file


# Load the latest versioned model file
try:
    joblib_file_path = find_latest_versioned_model(base_joblib_filename)
    model = load(joblib_file_path)
    print(f"Loaded model from {joblib_file_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Patch the missing attribute
# for estimator in model.estimators_:
#     if not hasattr(estimator, 'monotonic_cst'):
#         estimator.monotonic_cst = None

app = FastAPI()


class ScoringItem(BaseModel):
    """
    Model representing scoring parameters for prediction.
    """
    place: int
    catu: int
    sexe: int
    secu1: float
    year_acc: int
    victim_age: float
    catv: float
    obsm: float
    motor: float
    catr: int
    circ: float
    surf: float
    situ: float
    vma: float
    jour: int
    mois: int
    lum: int
    dep: int
    com: int
    agg_: int
    int_: int
    atm: float
    col: float
    lat: float
    long: float
    hour: int
    nb_victim: int
    nb_vehicules: int


@app.post('/predict')
async def predict(input_data: ScoringItem):
    """
    Endpoint for secure prediction based on scoring parameters.

    Args:
        item (ScoringItem): Input parameters for prediction.

    Returns:
        dict: Prediction result, can be 1 or 0 indicating shot made or missed.
    """
    # Create a DataFrame with the data of the request object
    df = pd.DataFrame([input_data.model_dump()])
    # Rename the columns to match the expected names
    
    # # Make a prediction with the loaded model
    # yhat = model.predict(df)

    # Predict the target variable for the test set
    probabilities = model.predict_proba(df)[:, 1] 
    yhat = (probabilities > 0.5).astype(int)
    # Return the prediction as an answer
    return {"prediction": int(yhat.item())}