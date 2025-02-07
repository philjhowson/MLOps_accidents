import argparse
import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
import sklearn
import os
import pandas as pd 
from sklearn import ensemble
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import json
import numpy as np

print(f"joblib version: {joblib.__version__}")

def train_rfc():

    X_train = pd.read_csv('data/preprocessed/X_train.csv')
    X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include = 'int').columns})
    X_test = pd.read_csv('data/preprocessed/X_test.csv')
    X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes(include = 'int').columns})
    y_train = pd.read_csv('data/preprocessed/y_train.csv')
    y_test = pd.read_csv('data/preprocessed/y_test.csv')
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    params = {'n_estimators' : [50, 100, 150, 200],
              'max_depth' : [5, 10, 15, None],
              'min_samples_split' : [2, 5, 10],
              'min_samples_leaf' : [1, 2, 5]
              }

    rf_classifier = ensemble.RandomForestClassifier(n_jobs = -1)
    grid_search = GridSearchCV(rf_classifier, params, cv = 3, scoring = 'f1')
    grid_search.fit(X_train, y_train)


    tracking_dir = 'mlruns/RandomForests'
    working_dir = os.getcwd()
    full_path = os.path.join(working_dir, tracking_dir)
    full_path = full_path.replace('\\', '/')

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    mlflow.set_tracking_uri(f"file:///{full_path}")
    mlflow.set_experiment('RandomForests')

    best_index = grid_search.best_index_
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_F1 = grid_search.cv_results_['mean_test_score'][best_index]

    with mlflow.start_run(run_name='RandomForests') as run:
        param_grid_json = json.dumps(grid_search.param_grid, indent=2)
        mlflow.log_param('param_grid', param_grid_json)
        mlflow.set_tag('param_grid_full', param_grid_json)

        for key, value in best_params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric('Training F1-Score', best_F1)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model,
            artifact_path='best_random_forests_params',
            signature=signature
        )

        mlflow.set_tag('best_model', 'True')
        mlflow.log_param('X_train length', X_train.shape[0])
        mlflow.log_param('Number of training features', X_train.shape[1])

        y_test_pred = best_model.predict(X_test)
        test_F1 = f1_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)

        mlflow.log_metric('Test F1-Score', test_F1)
        mlflow.log_metric('Test ROC-AUC', test_roc_auc)

        mlflow.end_run()
        
    print('Best Parameters:', grid_search.best_params_)
    print('Best F1-Score:', grid_search.best_score_)

    rf_classifier = grid_search.best_estimator_

    model_filename = 'models/best_random_forests.joblib'
    joblib.dump(rf_classifier, model_filename)
    print('Model trained and saved successfully.')

    scores = {'Training Score' : grid_search.best_score_}
    y_pred = rf_classifier.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    scores['Test Score'] = test_f1

    with open('metrics/RandomForests_scores.json', 'w') as f:
        json.dump(scores, f)

    print(f"Test Score: {test_f1}")
    print('Test Scores successfully saved.')

if __name__ == "__main__":

    train_rfc()
