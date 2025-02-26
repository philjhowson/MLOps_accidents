import sys
import os
import requests
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from mlflow import MlflowClient
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
import logging
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    return True
    
def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, ask if we want to create it.'''
    if not os.path.exists(folder_path):
        return True
    else:
        return False
    
def process_data(input_filepath_users, input_filepath_caract, input_filepath_places, input_filepath_veh, output_folderpath):
 
    #--Importing dataset
    df_users = pd.read_csv(input_filepath_users, sep = ";")
    df_caract = pd.read_csv(input_filepath_caract, sep = ";", header = 0, low_memory = False)
    df_places = pd.read_csv(input_filepath_places, sep = ";", encoding='utf-8')
    df_veh = pd.read_csv(input_filepath_veh, sep=";")

    # Change Accident_Id col to Num_Acc
    if 'Accident_Id' in df_caract.columns:
        df_caract.rename(columns={'Accident_Id': 'Num_Acc'}, inplace=True)

    #--Creating new columns
    nb_victim = pd.crosstab(df_users.Num_Acc, "count").reset_index()
    nb_vehicules = pd.crosstab(df_veh.Num_Acc, "count").reset_index()
    df_users["year_acc"] = df_users["Num_Acc"].astype(str).apply(lambda x : x[:4]).astype(int)
    df_users["victim_age"] = df_users["year_acc"]-df_users["an_nais"]
    for i in df_users["victim_age"] :
        if (i>120)|(i<0):
            df_users["victim_age"].replace(i,np.nan)
    df_caract["hour"] = df_caract["hrmn"].astype(str).apply(lambda x : x[:-3])
    df_caract.drop(['hrmn', 'an'], inplace=True, axis=1)
    df_users.drop(['an_nais'], inplace=True, axis=1)

    #--Replacing names 
    df_users.grav.replace([1,2,3,4], [1,3,4,2], inplace = True)
    df_caract.rename({"agg" : "agg_"},  inplace = True, axis = 1)
    #corse_replace = {"2A":"201", "2B":"202"}
    df_caract["dep"] = df_caract["dep"].str.replace("2A", "201")
    df_caract["dep"] = df_caract["dep"].str.replace("2B", "202")
    df_caract["com"] = df_caract["com"].str.replace("2A", "201")
    df_caract["com"] = df_caract["com"].str.replace("2B", "202")

    # Drop undefined values
    df_caract = df_caract[df_caract['com'] != 'N/C']

    #--Converting columns types
    df_caract[["dep","com", "hour"]] = df_caract[["dep","com", "hour"]].astype(int)

    dico_to_float = { 'lat': float, 'long':float}
    df_caract["lat"] = df_caract["lat"].str.replace(',', '.')
    df_caract["long"] = df_caract["long"].str.replace(',', '.')
    df_caract = df_caract.astype(dico_to_float)


    #--Grouping modalities 
    dico = {1:0, 2:1, 3:1, 4:1, 5:1, 6:1,7:1, 8:0, 9:0}
    df_caract["atm"] = df_caract["atm"].replace(dico)
    catv_value = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,30,31,32,33,34,35,36,37,38,39,40,41,42,43,50,60,80,99]
    catv_value_new = [0,1,1,2,1,1,6,2,5,5,5,5,5,4,4,4,4,4,3,3,4,4,1,1,1,1,1,6,6,3,3,3,3,1,1,1,1,1,0,0]
    df_veh['catv'].replace(catv_value, catv_value_new, inplace = True)

    #--Merging datasets 
    fusion1= df_users.merge(df_veh, on = ["Num_Acc","num_veh", "id_vehicule"], how="inner")
    fusion1 = fusion1.sort_values(by = "grav", ascending = False)
    fusion1 = fusion1.drop_duplicates(subset = ['Num_Acc'], keep="first")
    fusion2 = fusion1.merge(df_places, on = "Num_Acc", how = "left")
    df = fusion2.merge(df_caract, on = 'Num_Acc', how="left")

    #--Adding new columns
    df = df.merge(nb_victim, on = "Num_Acc", how = "inner")
    df.rename({"count" :"nb_victim"},axis = 1, inplace = True) 
    df = df.merge(nb_vehicules, on = "Num_Acc", how = "inner") 
    df.rename({"count" :"nb_vehicules"},axis = 1, inplace = True)

    #--Modification of the target variable  : 1 : prioritary // 0 : non-prioritary
    df['grav'].replace([2,3,4], [0,1,1], inplace = True)


    #--Replacing values -1 and 0 
    col_to_replace0_na = [ "trajet", "catv", "motor"]
    col_to_replace1_na = [ "trajet", "secu1", "catv", "obsm", "motor", "circ", "surf", "situ", "vma", "atm", "col"]
    df[col_to_replace1_na] = df[col_to_replace1_na].replace(-1, np.nan)
    df[col_to_replace0_na] = df[col_to_replace0_na].replace(0, np.nan)


    #--Dropping columns 
    list_to_drop = ['senc','larrout','actp', 'manv', 'choc', 'nbv', 'prof', 'plan', 'Num_Acc', 'id_vehicule', 'num_veh', 'pr', 'pr1','voie', 'trajet',"secu2", "secu3",'adr', 'v1', 'lartpc','occutc','v2','vosp','locp','etatp', 'infra', 'obs' ]
    df.drop(list_to_drop, axis=1, inplace=True)

    #--Dropping lines with NaN values
    col_to_drop_lines = ['catv', 'vma', 'secu1', 'obsm', 'atm']
    df = df.dropna(subset = col_to_drop_lines, axis=0)

    return df


def dag_import_raw_data():
    raw_data_relative_path="./data/raw" 
    filenames = ["caracteristiques-2021.csv", "lieux-2021.csv", "usagers-2021.csv", 
                "vehicules-2021.csv"]
    bucket_folder_url= "https://mlops-project-db.s3.eu-west-1.amazonaws.com/accidents/" 

    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    
    # download all the files
    for filename in filenames :
        input_file = os.path.join(bucket_folder_url,filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            object_url = input_file
            print(f'downloading {input_file} as {os.path.basename(output_file)}')
            response = requests.get(object_url)
            if response.status_code == 200:
                # Process the response content as needed
                content = response.text
                text_file = open(output_file, "wb")
                text_file.write(content.encode('utf-8'))
                text_file.close()
            else:
                print(f'Error accessing the object {input_file}:', response.status_code)

def dag_make_dataset():
    input_filepath = './data/raw/'
    input_filepath_users = os.path.join(input_filepath, "usagers-2021.csv")
    input_filepath_caract = os.path.join(input_filepath, "caracteristiques-2021.csv")
    input_filepath_places = os.path.join(input_filepath, "lieux-2021.csv")
    input_filepath_veh = os.path.join(input_filepath, "vehicules-2021.csv")
    output_filepath = './data/preprocessed/'
    
    # Call the main data processing function with the provided file paths
    df = process_data(input_filepath_users, input_filepath_caract, input_filepath_places, input_filepath_veh, output_filepath)
    if check_existing_folder(output_filepath):
        os.makedirs(output_filepath)
    df.to_csv(os.path.join(output_filepath, "training_data.csv"), index = False)

def dag_split_data():
    output_folderpath = './data/preprocessed/'
    df = pd.read_csv('./data/preprocessed/training_data.csv', index_col = False)
    target = df['grav']
    feats = df.drop(['grav'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state = 42)

    #--Filling NaN values
    col_to_fill_na = ["surf", "circ", "col", "motor"]
    X_train[col_to_fill_na] = X_train[col_to_fill_na].fillna(X_train[col_to_fill_na].mode().iloc[0])
    X_test[col_to_fill_na] = X_test[col_to_fill_na].fillna(X_train[col_to_fill_na].mode().iloc[0])

    # drop id_usager from train and test set
    # X_train.drop(['id_usager'], axis=1, inplace=True)
    # X_test.drop(['id_usager'], axis=1, inplace=True)

    # Create folder if necessary 
    if check_existing_folder(output_folderpath) :
        os.makedirs(output_folderpath)
    

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)

def dag_train_random_forests():
    print("Training Random Forests...")

    X_train = pd.read_csv('data/preprocessed/X_train.csv')
    X_test = pd.read_csv('data/preprocessed/X_test.csv')
    y_train = pd.read_csv('data/preprocessed/y_train.csv')
    y_test = pd.read_csv('data/preprocessed/y_test.csv')
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    #set parameters
    params = {'n_estimators' : [50, 100],# 150, 200],
              'max_depth' : [5, 10], # 15, None],
              'min_samples_split' : [2],# 5, 10],
              'min_samples_leaf' : [1],# 2, 5]
              }

    #initialize mlflow
    rf_classifier = ensemble.RandomForestClassifier(n_jobs = -1)
    grid_search = GridSearchCV(rf_classifier, params, cv = 3, scoring = 'f1')
    grid_search.fit(X_train, y_train)

    current_dir = os.getcwd()
    tracking_dir = os.path.join(current_dir, f"mlruns/RandomForests")
    MlflowClient(tracking_uri = f'file:///{os.path.abspath(tracking_dir)}')
    
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

# def dag_simulate_data():
#     # simulate_data()

with DAG(
    dag_id='training_pipeline',
    schedule_interval=None,
    tags=['training', 'french_accidents'],
    start_date=days_ago(2),
    catchup=False
) as dag:
    task_import_raw_data = PythonOperator(
        task_id='import_raw_data',
        python_callable=dag_import_raw_data
    )

    task_make_dataset = PythonOperator(
        task_id='make_dataset',
        python_callable=dag_make_dataset
    )

    task_split_data = PythonOperator(
        task_id='split_data',
        python_callable=dag_split_data
    )

    task_train_random_forests = PythonOperator(
        task_id='train_random_forests',
        python_callable=dag_train_random_forests
    )

    # task_simulate_data = PythonOperator(
    #     task_id='simulate_data',
    #     python_callable=dag_simulate_data
    # )

    task_import_raw_data >> task_make_dataset >> task_split_data >> task_train_random_forests