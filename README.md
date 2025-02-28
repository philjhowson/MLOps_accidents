Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "road accident". It's not perfect so feel free to make some modifications on it.

Project Organization
```
MLOps_accidents/
├── .dvc/
│   ├── config
|
├── data/
│   ├── preprocessed/
│   │   ├── training_data.csv.dvc
│   ├── raw/
│
├── metrics/
│   ├── RandomForests_scores.json.dvc
│
├── models/
│   ├── best_random_forests.joblib.dvc
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── app.py
│   │   ├── requirements.txt
│   ├── data/
│   │   ├── __init__.py
│   │   ├── check_structure.py
│   │   ├── import_raw_data.py
│   │   ├── make_dataset.py
│   │   ├── requirements.txt
│   │   ├── simulate_data.py
│   │   ├── split_dataset.py
│   │   ├── update_training_dataset.py
│   ├── docker/
│   │   ├── docker-compose.yaml
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.import_raw_data
│   │   ├── Dockerfile.make_dataset
│   │   ├── Dockerfile.simulate_data
│   │   ├── Dockerfile.split_dataset
│   │   ├── Dockerfile.train_random_forest
│   ├── test/
│   │   ├── test_api_endpoint.py
│
├── README.md
├── requirements.txt
├── simulate_airflow.sh
├── LICENSE
```

## Steps to follow 

1. Run `source simulate_airflow.sh` which is simulating an airflow DAG. You can use the docker compose or ingle python files by commenting the code. You must have already initialized DVC and GIT
2. Run `evidently ui --workspace ./datascientest-workspace/ --host 0.0.0.0 --port 8000` for Evidently
3. Run `mlflow ui` for MLFlow
4. Open the following addresses on your internetbrowser localhost:8888/redoc (API), localhost:5000 (MLFlow) and localhost:8000 (Evidently)

------------------------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
