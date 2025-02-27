# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import json
from typing import Any

MODEL_PATH = '../../../../models/best_random_forests.joblib'
model = joblib.load(MODEL_PATH)

app = FastAPI()

class DataFrameRequest(BaseModel):
    data: Any  # This will accept any data (which can be JSON-like structure)

@app.post("/predictions/")
async def process_dataframe(request: DataFrameRequest):
    # Convert the input data into a Pandas DataFrame
    df = pd.DataFrame(request.data)
    #print(df.head(5))
    result = model.predict(df)
    #print(result)
    result = {'predictions': result.tolist()}
    #result = pd.DataFrame(result)
    #result = result.to_dict()
    #return {"dataframe_head": result}
    return result
