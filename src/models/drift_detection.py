import pandas as pd
import joblib
import json
import requests
from prometheus_client import start_http_server, Gauge
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from update_model import update_model

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

    reference_data = pd.read_csv('data/preprocessed/reference_data.csv').drop(columns = ['Probability 0', 'Probability 1'])
    reference = reference_data.drop(columns = ['jour'])
    updated_data = pd.read_csv('data/preprocessed/updated_data.csv')
    updated = updated_data.drop(columns = ['jour'])

    column_mapping = ColumnMapping()
    column_mapping.target = 'grav'
    column_mapping.prediction = 'Predictions'
    column_mapping.numerical_features = reference.drop(columns = ['year_acc']).columns.tolist()

    drift_report = Report(metrics = [
        DataDriftPreset(), TargetDriftPreset()])

    years = [2022, 2023]
    months = range(1, 13)
    triggered = 0

    start_http_server(8000, addr = '0.0.0.0')

    for year in years:

        for month in months:

            if triggered == 0:
                ref = reference[reference['mois'] == month].drop(columns = ['mois', 'year_acc'])
            else:
                ref = data[data['mois'] == month].drop(columns = ['mois', 'year_acc'])
                
            current = updated[(updated['year_acc'] == year) &
                              (updated['mois'] == month)].drop(columns = ['mois', 'year_acc'])
            
            drift_report.run(reference_data = ref,
                             current_data = current,
                             column_mapping = column_mapping)

            drift_report.save_json(f'metrics/model_updates/drift_report_month_{month}_{year}.json')

            drift = drift_report.as_dict()

            data_drift_score = drift['metrics'][0]['result'].get('share_of_drifted_columns', None)
            target_drift_score = drift['metrics'][1]['result']['drift_by_columns']['grav'].get('drift_score', None)

            data_drift_guage.set(data_drift_score)
            target_drift_guage.set(target_drift_score)

            if data_drift_score > 0.15 or target_drift_score > 0.15:

                message = f"Significant drift detected for month {month} year {year}. Retraining model."
                print(message)
                #send_slack_notification(message, slack_webhook_url)

                triggered = 1
                drift_report.save_html(f'metrics/model_updates/drift_report_month_{month}_{year}.html')
                update_model(year, month)

                data = pd.read_csv('data/preprocessed/updated_reference_data.csv')

if __name__ == '__main__':
    drift_detection()
