import argparse
import mlflow
from mlflow.models.signature import infer_signature
import sklearn
import os
import pandas as pd 
from sklearn import ensemble
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import joblib
import json
import numpy as np

print(f"joblib version: {joblib.__version__}")

def train_rfc():

    X_train = pd.read_csv('data/preprocessed/X_train.csv')
    X_test = pd.read_csv('data/preprocessed/X_test.csv')
    y_train = pd.read_csv('data/preprocessed/y_train.csv')
    y_test = pd.read_csv('data/preprocessed/y_test.csv')
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    #set parameters

    params = {'n_estimators' : [50, 100, 150, 200],
              'max_depth' : [5, 10, 15, None],
              'min_samples_split' : [2, 5, 10],
              'min_samples_leaf' : [1, 2, 5]
              }

    #initialize mlflow

    rf_classifier = ensemble.RandomForestClassifier(n_jobs = -1)
    grid_search = GridSearchCV(rf_classifier, params, cv = 3, scoring = 'f1')
    grid_search.fit(X_train, y_train)

    current_dir = os.getcwd()
    tracking_dir = os.path.join(current_dir, f"mlruns/RandomForests")
    mlflow.set_tracking_uri(f'file:///{os.path.abspath(tracking_dir)}')
    
    mlflow.set_experiment("RandomForests")

    for index, params in enumerate(grid_search.cv_results_["params"]):
        with mlflow.start_run(run_name = f"RandomForests"):
            F1 = grid_search.cv_results_["mean_test_score"][index]

            mlflow.log_params(params)
            mlflow.log_metric("F1-Score", F1)

            if params == grid_search.best_params_:
                signature = infer_signature(X_train, grid_search.best_estimator_.predict(X_train))
                mlflow.sklearn.log_model(grid_search.best_estimator_, "best_random_forests_params", signature = signature)

    mlflow.end_run()

    print("Best Parameters:", grid_search.best_params_)
    print("Best F1-Score:", grid_search.best_score_)

    rf_classifier = grid_search.best_estimator_

    #--Save the trained model to a file
    model_filename = f'models/best_random_forests.joblib'
    joblib.dump(rf_classifier, model_filename)
    print("Model trained and saved successfully.")

    scores = {'Training Score' : grid_search.best_score_}
    y_pred = rf_classifier.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    scores['Test Score'] = test_f1

    with open(f"metrics/RandomForests_scores.json", 'w') as f:
        json.dump(scores, f)

    print(f"Test Score: {test_f1}")
    print('Test Scores successfully saved.')

if __name__ == "__main__":

    train_rfc()


