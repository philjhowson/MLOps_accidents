import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
import pandas as pd
import joblib
import src.models.utility_evidently as utility

def analyze_data_drift(input_filepath, month, year):
    target = 'grav'
    # left com, long, lat, jour, mois, year
    numerical_features = ['victim_age','vma','nb_victim','nb_vehicules','hour']
    categorical_features = ['place', 'catu', 'sexe','catr','lum','dep','agg_','int','secu1','catv','obsm','motor','circ','surf','situ','atm','col'] 
    
    reference_data = pd.read_csv(os.path.join(input_filepath, 'training_data.csv'))
    current_data = pd.read_csv(os.path.join(input_filepath, 'simulated_data.csv'))
    current_data = current_data[(current_data['mois'] == month) & (current_data['year_acc'] == year)]   

    # Perform column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = None
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Generate reference and current data with numerical values only
    current_data = current_data[numerical_features + categorical_features]
    reference_data = reference_data[numerical_features + categorical_features]

    # Initialize the data drift report with the default data drift preset
    data_drift_report = Report(metrics=[
        DataDriftPreset(
            drift_share='0.2'
        ),
        ],
        tags=[f'{year}-{month}']
    )

    # Run the data drift report using the reference data
    data_drift_report.run(reference_data=reference_data.sort_index(), 
                                    current_data=current_data.sort_index(),
                                    column_mapping=column_mapping)

    report_dict = data_drift_report.as_dict()
    dataset_drift_status = report_dict["metrics"][0]["result"]["dataset_drift"]
    print(f"Dataset Drift Detected: {dataset_drift_status}")

    return data_drift_report

def main(input_filepath, month, year):
    report = analyze_data_drift(input_filepath, month, year)
    utility.add_report(report, '06_data_drift')

if __name__ == '__main__':
    for month in range(1, 13):
        report = analyze_data_drift('data/preprocessed', month)
        utility.add_report(report, '06_data_drift')