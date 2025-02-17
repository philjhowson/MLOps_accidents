import pandas as pd
import numpy as np
import json
import joblib
import os
import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

def update_model(year, month):

    X = pd.read_csv('data/preprocessed/reference_data.csv').drop(columns = ['Probability 0', 'Probability 1'])
    X_ = pd.read_csv('data/preprocessed/updated_data.csv')

    if year == 2022:
        X1 = X_[(X_['year_acc'] == year) & (X_['mois'] <= month)]
        X = pd.concat([X, X1], axis = 0)
        
    if year == 2023:
        X1 = X_[X_['year_acc'] == 2022]
        X2 = X_[(X_['year_acc'] == 2023) & (X_['mois'] <= month)]
        X = pd.concat([X, X1, X2])

    y = X['grav']
    X.drop(columns = ['Predictions', 'grav'], inplace = True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                        random_state = 42)

    params = {'n_estimators' : [50, 100, 150, 200],
              'max_depth' : [5, 10, 15, None],
              'min_samples_split' : [2, 5, 10],
              'min_samples_leaf' : [1, 2, 5]
              }

    rf_classifier = RandomForestClassifier(n_jobs = -1)
    grid_search = GridSearchCV(rf_classifier, params, cv = 3, scoring = 'f1')
    grid_search.fit(X_train, y_train)

    tracking_dir = 'mlruns/UpdatedRandomForests'
    working_dir = os.getcwd()
    full_path = os.path.join(working_dir, tracking_dir)
    full_path = full_path.replace('\\', '/')

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    mlflow.set_tracking_uri(f"file:///{full_path}")
    mlflow.set_experiment('UpdatedRandomForests')

    best_index = grid_search.best_index_
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_F1 = grid_search.cv_results_['mean_test_score'][best_index]

    with mlflow.start_run(run_name='UpdatedRandomForests'):
        param_grid_json = json.dumps(grid_search.param_grid, indent=2)
        mlflow.log_param('param_grid', param_grid_json)
        mlflow.set_tag('param_grid_full', param_grid_json)

        for key, value in best_params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric('Training F1-Score', best_F1)

        signature = infer_signature(X_train, best_model.predict(X_train))

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

    model_filename = 'models/updated_random_forests.joblib'
    joblib.dump(rf_classifier, model_filename)
    print('Model updated and saved successfully.')

    scores = {'Training Score' : grid_search.best_score_}
    y_pred = rf_classifier.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    scores['Test Score'] = test_f1

    y_pred = rf_classifier.predict(X)
    X['Predictions'] = y_pred
    X['grav'] = y

    X.to_csv('data/preprocessed/updated_reference_data.csv', index = False)

    with open('metrics/updated_RandomForests_scores.json', 'w') as f:
        json.dump(scores, f)

    print(f"Test Score: {test_f1}")
    print('Test Scores successfully saved.')
    
