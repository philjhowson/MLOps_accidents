import pandas as pd
import json
import joblib
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import ClassificationPreset

def classification_reports():

    data = pd.read_csv('data/preprocessed/X_train.csv')
    data_ = pd.read_csv('data/preprocessed/X_test.csv')
    data = pd.concat([data, data_], axis = 0)

    target = pd.read_csv('data/preprocessed/y_train.csv')
    target_ = pd.read_csv('data/preprocessed/y_test.csv')
    target = pd.concat([target, target_], axis = 0)

    updated_data = pd.read_csv('data/preprocessed/features_2022_2023.csv')
    updated_target = pd.read_csv('data/preprocessed/targets_2022-2023.csv')

    with open('models/best_random_forests.joblib', 'rb') as f:
        model = joblib.load(f)

    probabilities = model.predict_proba(data)
    preds = model.predict(data)
    data['Predictions'] = preds
    data['Probability 0'], data['Probability 1'] = probabilities[:, 0], probabilities[:, 1]
    reference_data = pd.concat([data, target], axis = 1)

    probabilities = model.predict_proba(updated_data)
    preds = model.predict(updated_data)
    updated_data['Predictions'], updated_data['grav'] = preds, updated_target['grav']

    column_mapping = ColumnMapping()
    column_mapping.target = 'grav'
    column_mapping.prediction = 'Predictions'
    column_mapping.prediction_proba = ['Probability 0', 'Probability 1']
    column_mapping.numerical_features = reference_data.columns.tolist()

    classification_performance_report = Report(metrics = [
        ClassificationPreset()])

    classification_performance_report.run(reference_data = None,
                                          current_data = reference_data,
                                          column_mapping = column_mapping)

    classification_performance_report.save_json('metrics/reference_data_report.json')
    classification_performance_report.save_html('metrics/reference_data_report.html')

    reference_data.to_csv('data/preprocessed/reference_data.csv', index = False)
    updated_data.to_csv('data/preprocessed/updated_data.csv', index = False)

    print('Model report complete! Reports, reference, and updated data saved.')

if __name__ == '__main__':
    classification_reports()
