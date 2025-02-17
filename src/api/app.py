from fastapi import FastAPI
import joblib
import os
from pydantic import BaseModel
import pandas as pd

# Define model path and load it
#MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_random_forests.joblib"))
MODEL_PATH = 'best_random_forests.joblib'
model = joblib.load(MODEL_PATH)

class AccidentData(BaseModel):
    place: int
    catu: int
    sexe: int
    secu1: float
    year_acc: int
    victim_age: int
    catv: int
    obsm: int
    motor: int
    catr: int
    circ: int
    surf: int
    situ: int
    vma: int
    jour: int
    mois: int
    lum: int
    dep: int
    com: int
    agg_: int
    int: int
    atm: int
    col: int
    lat: float
    long: float
    hour: int
    nb_victim: int
    nb_vehicules: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    'place': 10,
                    'catu': 3,
                    'sexe': 1,
                    'secu1': 0.0,
                    'year_acc': 2021,
                    'victim_age': 60,
                    'catv': 2,
                    'obsm': 1,
                    'motor': 1,
                    'catr': 3,
                    'circ': 2,
                    'surf': 1,
                    'situ': 1,
                    'vma': 50,
                    'jour': 7,
                    'mois': 12,
                    'lum': 5,
                    'dep': 77,
                    'com': 77317,
                    'agg_': 2,
                    'int': 1,
                    'atm': 0,
                    'col': 6,
                    'lat': 48.6,
                    'long': 2.89,
                    'hour': 17,
                    'nb_victim': 2,
                    'nb_vehicules': 1
                }
            ]
        }
    }

class Prediction(BaseModel):
    grav: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "grav": 0
                }
            ]
        }
    }

app = FastAPI()

@app.get("/predict/", summary="Predict the severity of an accident")
def predict_seriousness_accident(features: AccidentData) -> Prediction:
    features = pd.DataFrame.from_dict(features.dict(), orient='index').T
    prediction = model.predict(features)
    return {'grav': int(list(prediction)[0])}
