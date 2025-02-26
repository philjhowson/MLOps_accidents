import pandas as pd
import json
import joblib
import os
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import ClassificationPreset

def classification_reports(year = None, month = None):

    if not os.path.exists('metrics/classification_reports/original'):
        os.makedirs('metrics/classification_reports/original')

    if not year or not month:
        data = pd.read_csv('data/preprocessed/X_train.csv')
        data_ = pd.read_csv('data/preprocessed/X_test.csv')
        data = pd.concat([data, data_], axis = 0)

        features = data.columns.to_list()

        target = pd.read_csv('data/preprocessed/y_train.csv')
        target_ = pd.read_csv('data/preprocessed/y_test.csv')
        target = pd.concat([target, target_], axis = 0)

        with open('models/best_random_forests.joblib', 'rb') as f:
            model = joblib.load(f)

        probabilities = model.predict_proba(data)
        preds = model.predict(data)
        data['Predictions'] = preds
        data['Probability 0'], data['Probability 1'] = probabilities[:, 0], probabilities[:, 1]
        reference_data = pd.concat([data, target], axis = 1)
        reference_data.to_csv('data/preprocessed/reference_data.csv', index = False)

    if year and month:

        if not os.path.exists('metrics/classification_reports/updates'):
            os.makedirs('metrics/classification_reports/updates')        
        
        previous_data = pd.read_csv('data/preprocessed/reference_data.csv')
        previous_data.drop(columns = ['Predictions', 'Probability 0', 'Probability 1'], inplace = True)
        updated_data = pd.read_csv('data/preprocessed/updated_data.csv')
        updated_data = updated_data[(updated_data['mois'] <= month) &
                                    (updated_data['year_acc'] <= year)]
        reference_data = pd.concat([previous_data, updated_data], axis = 0)
        target = reference_data['grav']
        reference_data.drop('grav', inplace = True)

        with open('models/updated_random_forests.joblib', 'rb') as f:
            model = joblib.load(f)

        probabilities = model.predict_proba(data)
        preds = model.predict(data)
        data['Predictions'] = preds
        data['Probability 0'], data['Probability 1'] = probabilities[:, 0], probabilities[:, 1]
        reference_data = pd.concat([data, target], axis = 1)
        reference_data.to_csv('data/preprocessed/updated_reference_data.csv', index = False)
         
    column_mapping = ColumnMapping()
    column_mapping.target = 'grav'
    column_mapping.prediction = 'Predictions'
    column_mapping.prediction_proba = ['Probability 0', 'Probability 1']
    column_mapping.numerical_features = features

    classification_performance_report = Report(metrics = [
        ClassificationPreset()])

    classification_performance_report.run(reference_data = None,
                                          current_data = reference_data,
                                          column_mapping = column_mapping)

    if not year or not month:
        classification_performance_report.save_json('metrics/classification_reports/original/reference_classification_report.json')
        classification_performance_report.save_html('metrics/classification_reports/original/reference_classification_report.html')
        print('Classification report complete! Reports, reference, and updated data saved.')

    if year and month:
        classification_performance_report.save_json(f'metrics/classification_reports/updates/updated_classification_data_report_{year}_{month}.json')
        classification_performance_report.save_html(f'metrics/classification_reports/updates/updated_classification_data_report_{year}_{month}.html')
        print('Updated classification report complete! Reports and reference data saved.')
       

if __name__ == '__main__':
    classification_reports(year = None, month = None)
