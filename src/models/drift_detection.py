import pandas as pd
import joblib
import json
import requests
import os
from prometheus_client import start_http_server, Gauge
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from update_model import update_model
from classification_report import classification_reports
from pathlib import Path

def drift_detection():

    slack_webhook_url = "https://hooks.slack.com/services/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"

    def send_slack_notification(message, webhook_url):
        payload = {"text": message}
        response = requests.post(webhook_url, json = payload)

        if response.status_code == 200:
            print("Slack notification sent successfully.")

        else:
            print(f"Failed to send notification. Status code: {response.status_code}")

    data_drift_guage = Gauge('data_drift_share', 'Share of Columns with Data Drift')
    target_drift_guage = Gauge('target_drift_share', 'Target Drift Score')

    ref_data = Path('data/preprocessed/updated_reference_data.csv')
    if ref_data.is_file():
        reference = pd.read_csv(ref_data).drop(columns = ['Probability 0', 'Probability 1'])
    else:
        reference = pd.read_csv('data/preprocessed/reference_data.csv').drop(columns = ['Probability 0', 'Probability 1'])

    year = reference['year_acc'].max()
    month = reference[reference['year_acc'] == year]
    month = month['mois'].max() + 1

    if month > 12:
        year += 1
        month = 1
        
    if year == 2024:
        return print('Target and Data Drift performed for all available data.')

    features = pd.read_csv('data/preprocessed/features_2022-2023.csv')
    targets = pd.read_csv('data/preprocessed/targets_2022-2023.csv')
    
    model = Path('models/updated_random_forests.joblib')
    if model.is_file():
        with open(model, 'rb') as f:
            model = joblib.load(f)
    else:
        with open('models/best_random_forests.joblib', 'rb') as f:
            model = joblib.load(f)

    current_features = features[(features['mois'] == month) & (features['year_acc'] == year)]   
    pred = model.predict(current_features)
    current_features['Predictions'] = pred
    matching_indices = targets.index.isin(current_features.index)
    targets = targets[matching_indices]
    current = pd.concat([current_features, targets], axis = 1).drop(columns = ['mois', 'year_acc'])    
    ref = reference[reference['mois'] == month].drop(columns = ['mois', 'year_acc'])

    column_mapping = ColumnMapping()
    column_mapping.target = 'grav'
    column_mapping.prediction = 'Predictions'
    column_mapping.numerical_features = reference.drop(columns = ['year_acc']).columns.tolist()

    drift_report = Report(metrics = [
        DataDriftPreset(), TargetDriftPreset()])

    try:
        start_http_server(8000, addr="0.0.0.0")
        print("Prometheus metrics server started.")

    except OSError as e:
        if "Address already in use" in str(e):
            pass
        else:
            raise
        
    drift_report.run(reference_data = ref,
                     current_data = current,
                     column_mapping = column_mapping)

    if not os.path.exists('metrics/model_updates/'):
        os.mkdir('metrics/model_updates/')
    
    drift_report.save_json(f'metrics/model_updates/drift_report_month_{month}_{year}.json')
    drift = drift_report.as_dict()

    current_reference_data = pd.concat([ref, current], axis = 0)
    current_reference_data.to_csv('data/preprocessed/updated_reference_data.csv')

    data_drift_score = drift['metrics'][0]['result'].get('share_of_drifted_columns', None)
    target_drift_score = drift['metrics'][1]['result']['drift_by_columns']['grav'].get('drift_score', None)

    data_drift_guage.set(data_drift_score)
    target_drift_guage.set(target_drift_score)

    if data_drift_score > 0.15 or target_drift_score > 0.15:

        message = f"Significant drift detected for month {month} year {year}. Retraining model."
        print(message)
        #send_slack_notification(message, slack_webhook_url)
        drift_report.save_html(f'metrics/model_updates/drift_report_month_{month}_{year}.html')
        update_model(year, month)
        classification_reports(year, month)

if __name__ == '__main__':
    drift_detection()
