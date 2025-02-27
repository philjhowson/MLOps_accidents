import requests
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import joblib
import logging
import numpy as np

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, DataDriftTable
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

import datetime

#from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg

from airflow.models import Variable
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

num_features = ['year_acc', 'victim_age', 'vma', 'dep', 'com', 'lat', 'long' ]
cat_features = ['place', 'catu', 'sexe', 'secu1', 'catv', 'obsm', 'motor', 'catr', 'circ', 'surf', 'situ', 'jour', 'mois', 'lum', 'agg_', 'int', 'atm', 'col', 'hour', 'nb_victim', 'nb_vehicules' ]


# this function queries the current data for a given month of a year
# in a real implementation, this query should be change to a query of
# a database containing all current data
def current_dataset_month(year: int, month: int) -> pd.DataFrame:
    features = pd.read_csv('/data/preprocessed/features_2022-2023.csv')
    targets = pd.read_csv('/data/preprocessed/targets_2022-2023.csv')

    df = pd.concat([features, targets], axis=1)
    df.index = pd.to_datetime(df[['year_acc', 'mois', 'jour', 'hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
    df = df.sort_index(ascending=True)
    df.dropna(axis=0, how='any', inplace=True)
    return df[(df.index.year == year) & (df.index.month == month)]

# this function queries the reference/training data for a given month of a year
# in a real implementation, this query should be change to a query of
# a database containing all training data
def reference_dataset_month(year: int, month: int) -> pd.DataFrame:
    df = pd.read_csv('/data/preprocessed/test_reference_data.csv')
    df = df.drop(columns = ['Probability 0', 'Probability 1', 'Predictions'])
    df.index = pd.to_datetime(df[['year_acc', 'mois', 'jour', 'hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
    df = df.sort_index(ascending=True)
    df.dropna(axis=0, how='any', inplace=True)
    return df[(df.index.year == year) & (df.index.month == month)]

# this function queries the prediction from model API for a dataset
def make_predictions_dataset_api(df: pd.DataFrame) -> pd.DataFrame:
    df_json = df.drop(columns=['grav']).to_dict(orient='records')
    #url = "http://host.docker.internal:8000/predictions/"
    url = "http://172.17.0.1:8000/predictions/"
    #url = "http://127.0.0.1:8000/predictions/"
    response = requests.post(url, json={"data": df_json})
    df_pred = pd.DataFrame(response.json())
    df_pred.index = df.index 
    df_return = pd.concat([df, df_pred], axis=1)
    return df_return
# this function calculates the metrics for the drift monitoring
def evaluate_dataset(df: pd.DataFrame) :
    mse = mean_squared_error(df['grav'], df['predictions'])
    mae = mean_absolute_error(df['grav'], df['predictions'])
    r2 = r2_score(df['grav'], df['predictions'])
    f1 = f1_score(df['grav'].tolist(), df['predictions'].tolist(), average = 'weighted')
    metrics = { 'mse': mse, 'mae': mae, 'r2': r2, 'f1': f1 }
    return metrics

# this function generates a drift report for the features and the target variables
def generate_report(reference_data, current_data, num_features, cat_features):
    # Define the column mapping for the Evidently report
    # This includes the prediction column, numerical features, and categorical features
    column_mapping = ColumnMapping(
        target='grav',
        prediction='predictions',
        numerical_features=num_features,
        categorical_features=cat_features
    )
    # Initialize the Evidently report with the desired metrics
    # In this case, we're using the ColumnDriftMetric for the 'prediction' column,
    # the DatasetDriftMetric to measure drift across the entire dataset,
    # and the DatasetMissingValuesMetric to measure the proportion of missing values
    report = Report(metrics=[
        ColumnDriftMetric(column_name='predictions'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DataDriftTable(),
        TargetDriftPreset(),
        DataDriftPreset()
    ])
    # Run the report on the training and validation data
    # The training data is used as the reference data, and the validation data is the current data
    report.run(reference_data=reference_data.reset_index(drop=True),
                current_data=current_data.reset_index(drop=True),
                column_mapping=column_mapping)
    # Return the generated report
    return report

# this will insert a new entry into the postgreSQL database only if the exact
# same entry does not exist so far
def insert_record(curr, next_month, df_jan_metric_current, data_drift_score, target_drift_score):
    # Check if the record already exists based on the timestamp (or other fields if needed)
    curr.execute("""
        SELECT 1 
        FROM MLOps_accidents 
        WHERE timestamp = %s
    """, (next_month,))
    
    # If a row exists with the same timestamp, don't insert
    if curr.fetchone():
        print("Record with this timestamp already exists. Skipping insert.")
    else:
        # Insert the record if it does not exist
        curr.execute("""
            INSERT INTO MLOps_accidents (timestamp, f1_score, data_drift_score, target_drift_score) 
            VALUES (%s, %s, %s, %s)
        """, (next_month, df_jan_metric_current['f1'], data_drift_score, target_drift_score))
        print("Record inserted successfully.")

# this function will accumulate metrics for the drift monitoring and store them in a postgreSQL databas
def calculate_metrics_postgresql(year: int, month: int, curr):
    print('start making month', month)
    df_jan_current = current_dataset_month(year, month)
    #df_jan_pred_current = make_predictions_dataset(df_jan_current)
    df_jan_pred_current = make_predictions_dataset_api(df_jan_current)
    df_jan_metric_current = evaluate_dataset(df_jan_pred_current)
    print(df_jan_metric_current) 

    df_jan_reference = reference_dataset_month(2021, month)
    df_jan_pred_reference = make_predictions_dataset_api(df_jan_reference)
    df_jan_metric_reference = evaluate_dataset(df_jan_pred_reference)
    print(df_jan_metric_reference)
    print('end making month')

    # logging.info("generating report...")
    # report = generate_report(df_jan_pred_reference, df_jan_pred_current, num_features, cat_features)
    # logging.info("Report generated successfully.")
    # result = report.as_dict()
    # print("Drift score of the prediction column: ", result['metrics'][0]['result']['drift_score'])
    # print("Number of drifted columns: ", result['metrics'][1]['result']['number_of_drifted_columns'])
    # print("Share of missing values: ", result['metrics'][2]['result']['current']['share_of_missing_values'])

    column_mapping = ColumnMapping()
    column_mapping.target = 'grav'
    column_mapping.prediction = 'predictions'
    column_mapping.numerical_features = num_features
    column_mapping.categorical_features = cat_features

    drift_report = Report(metrics = [
        DataDriftPreset(), TargetDriftPreset()])

    drift_report.run(reference_data = df_jan_pred_reference.reset_index(drop=True),
                    current_data = df_jan_pred_current.reset_index(drop=True),
                    column_mapping = column_mapping)

    drift = drift_report.as_dict()

    data_drift_score = drift['metrics'][0]['result'].get('share_of_drifted_columns', None)
    print('data_drift_score', data_drift_score)
    target_drift_score = drift['metrics'][1]['result']['drift_by_columns']['grav'].get('drift_score', None)
    print('target_drift_score', target_drift_score)

    #report.show(mode='inline')
    #drift_report.save_html('/drift_reports/drift_report_' + str(year) + '_' + str(month) + '.html')


    date_str = f'{year}-{month}-01'
    current_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')

    # Add one month using relativedelta to get the first day of the next month
    next_month = current_date + relativedelta(months=1)

    insert_record(curr, next_month, df_jan_metric_current, data_drift_score, target_drift_score)


create_table_statement = """
create table if not exists MLOps_accidents (
    timestamp timestamp,
    f1_score float,
    data_drift_score float,
    target_drift_score float
)
"""
# setup the database
def prep_db():
    """
    Prepare the PostgreSQL database by creating the necessary database and table if they do not exist.
    """
    # Connect to PostgreSQL server
    with psycopg.connect("host=172.17.0.1 port=5432 user=postgres password=example", autocommit=True) as conn:
        # Check if the 'test' database exists, create it if not
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        
        # Connect to the 'test' database
        with psycopg.connect("host=172.17.0.1 port=5432 dbname=test user=postgres password=example") as conn:
            # Check if the table MLOps_accidents exists
            res = conn.execute("SELECT to_regclass('MLOps_accidents')")
            if res.fetchone()[0] is None:
                # Create the 'MLOps_accidents' table if it does not exist
                conn.execute(create_table_statement)
                print("Table MLOps_accidents created.")
            else:
                print("Table MLOps_accidents already exists. Using the existing table.")

# function the gets triggered by the task that runs the metric calculation and storage in a database
def start_calculating_metric(year, month):
    # Prepare the PostgreSQL database
    prep_db()
    # Connect to the 'test' database
    with psycopg.connect("host=172.17.0.1 port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        print('year', year, 'month', month)
        # Create a cursor for executing SQL queries
        with conn.cursor() as curr:
            # Calculate and insert dummy metrics into the database
            calculate_metrics_postgresql(year, month, curr)

        logging.info("data sent")

# task that gets triggered every month to calculate the metrics
def task_01_calculate_metric(task_instance):
    # the counter hier is used to simulate increasing month
    # if this is running as running is monitor, the current year and month should
    # sent by airflow to evaluate the current month
    counter = Variable.get("month_counter", default_var=None)
    if counter is None:
        counter = 0  # Initialize the counter to 0 if not already set
    else:
        counter = int(counter)
    print('counter', counter)
    start_year = 2022
    start_month = 1
    date_str = f'{start_year}-{start_month}-01'
    start_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    current_date = start_date + relativedelta(months=counter)
    print('current_date.year', current_date.year)
    print('current_date.month', current_date.month)
    start_calculating_metric(current_date.year, current_date.month)
    Variable.set("month_counter", str(counter+1))

with DAG(
    dag_id='MLOps_accidents',
    tags=['MLOps_accidents', 'datascientest'],
    catchup=False,
    # the schedule '*/5 * * * *' (every 5 min) was set to simulate the 24 month of 2022 and 2023
    # to monitor every past month it should be changed to '0 0 1 * *' (every first day of the month)
    schedule_interval='*/5 * * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
        }
) as dag:
    task_01 = PythonOperator(
        task_id='task_01_query_raw_data',
        python_callable=task_01_calculate_metric
    )
    task_01