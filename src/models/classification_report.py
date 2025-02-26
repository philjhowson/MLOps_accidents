import pandas as pd
import json
import joblib
import os
from pathlib import Path
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import ClassificationPreset

def classification_report():

    if not os.path.exists('metrics/classification_reports/original'):
        os.makedirs('metrics/classification_reports/original')

    data = pd.read_csv('data/preprocessed/X_train.csv')
    indices = len(data)
    data_ = pd.read_csv('data/preprocessed/X_test.csv')
    data = pd.concat([data, data_], axis = 0)

    features = data.columns.to_list()
    num_features = ['year_acc', 'victim_age', 'vma', 'dep', 'com', 'lat', 'long']
    cat_features = [item for item in features if item not in num_features]

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

    reference_data = reference_data[indices:]
    reference_data.to_csv('data/preprocessed/test_reference_data.csv', index = False)

    column_mapping = ColumnMapping()
    column_mapping.target = 'grav'
    column_mapping.prediction = 'Probability 1'
    column_mapping.numerical_features = num_features
    column_mapping.categorical_features = cat_features

    classification_performance_report = Report(metrics = [
        ClassificationPreset()])

    classification_performance_report.run(reference_data = None,
                                          current_data = reference_data,
                                          column_mapping = column_mapping)

    classification_performance_report.save_json('metrics/classification_reports/original/reference_classification_report.json')
    classification_performance_report.save_html('metrics/classification_reports/original/reference_classification_report.html')
    print('Classification report complete! Reports, reference, and updated data saved.')       

def drift_classification_report(year = None, month = None):

    if not os.path.exists('metrics/classification_reports/updates'):
        os.makedirs('metrics/classification_reports/updates')        
    
    reference_data = pd.read_csv('data/preprocessed/test_reference_data.csv')
    current_data = pd.read_csv('data/preprocessed/updated_reference_data.csv')
    current_data = current_data[(current_data['year_acc'] == year) & (current_data['mois'] == month)]
    features = [col for col in current_data.columns.to_list() if col not in ['grav', 'Predictions', 'Probability 0', 'Probability 1']]
    num_features = ['year_acc', 'victim_age', 'vma', 'dep', 'com', 'lat', 'long']
    cat_features = [item for item in features if item not in num_features]

    column_mapping = ColumnMapping(target = 'grav',
                                   prediction = 'Probability 1',
                                   numerical_features = num_features,
                                   categorical_features = cat_features)

    classification_performance_report = Report(metrics = [
        ClassificationPreset()])

    classification_performance_report.run(current_data = current_data,
                                          reference_data = reference_data,
                                          column_mapping = column_mapping)

 
    classification_performance_report.save_json(f'metrics/classification_reports/updates/updated_classification_data_report_{month}_{year}.json')
    report_dict = classification_performance_report.as_dict()

    current_roc = report_dict['metrics'][0]['result']['current'].get('roc_auc', None)
    ref_roc = report_dict['metrics'][0]['result']['reference'].get('roc_auc', None)

    roc_diff = ref_roc - current_roc

    if roc_diff > 0.07:
        classification_performance_report.save_html(f'metrics/classification_reports/updates/updated_classification_data_report_{month}_{year}.html')

    print(f"Updated classification report complete for month {month} year {year}! Reports and reference data saved.")

    return current_roc, roc_diff

def updated_classification_report(year = None, month = None):
        
    if not os.path.exists('metrics/classification_reports/updates'):
        os.makedirs('metrics/classification_reports/updates')

    reference_data = pd.read_csv('data/preprocessed/updated_reference_data.csv').drop(columns = ['Predictions', 'Probability 0', 'Probability 1'])

    target = reference_data['grav']
    reference_data.drop(columns = ['grav'], inplace = True)

    features = reference_data.columns.to_list()
    num_features = ['year_acc', 'victim_age', 'vma', 'dep', 'com', 'lat', 'long']
    cat_features = [item for item in features if item not in num_features]

    model = Path('models/updated_random_forests.joblib')
    with open(model, 'rb') as f:
            model = joblib.load(f)

    probabilities = model.predict_proba(reference_data)
    preds = model.predict(reference_data)
    reference_data['Predictions'] = preds
    reference_data['Probability 0'], reference_data['Probability 1'] = probabilities[:, 0], probabilities[:, 1]
    reference_data = pd.concat([reference_data, target], axis = 1)
         
    column_mapping = ColumnMapping()
    column_mapping.target = 'grav'
    column_mapping.prediction = 'Probability 1'
    column_mapping.numerical_features = num_features
    column_mapping.categorical_features = cat_features


    classification_performance_report = Report(metrics = [
        ClassificationPreset()])

    classification_performance_report.run(reference_data = None,
                                          current_data = reference_data,
                                          column_mapping = column_mapping)

    classification_performance_report.save_json(f'metrics/classification_reports/updates/updated_classification_data_report_{year}_{month}.json')
    classification_performance_report.save_html(f'metrics/classification_reports/updates/updated_classification_data_report_{year}_{month}.html')
    print(f"Updated classification report complete for month {month} year {year}! Reports and reference data saved.")
       
if __name__ == '__main__':
    classification_report()