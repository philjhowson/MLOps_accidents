from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from models.drift_detection import drift_detection

default_args = {
    'owner' : 'MLOps_accidents',
    'depends_on_past' : False,
    'start_date' : datetime(2023, 1, 1),
    'retries' : 1}

with DAG('drift_detection', default_args = default_args,
         schedule = '/5 * * * *', catchup = False) as dag:

    #Task: Perform Drift Detection
    drift_detection_task = PythonOperator(
        task_id = 'drift_detection',
        python_callable = drift_detection)
