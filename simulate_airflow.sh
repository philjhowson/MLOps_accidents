#!/bin/sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

docker-compose -f src/docker/docker-compose.yml up --build

#python3 src/data/import_raw_data.py
#python3 src/data/make_dataset.py
#python3 src/data/split_dataset.py
#python3 src/models/train_random_forests.py
#python3 src/data/simulate_data.py

dvc add data/preprocessed/training_data.csv metrics/RandomForests_scores.json models/best_random_forests.joblib
git add data/preprocessed/training_data.csv.dvc metrics/RandomForests_scores.json.dvc models/best_random_forests.joblib.dvc
git commit -m "track the created data tables with dvc"
git push origin HEAD
dvc push -r origin data/preprocessed/training_data.csv metrics/RandomForests_scores.json models/best_random_forests.joblib
echo "Tracked files with DVC"

docker build -f src/docker/Dockerfile.api -t api:latest .
docker run -p 8888:8888 -d api


for year in 2022; do
    for month in {1..12}; do
        echo "Updating dataset for $year-$month"
        python -c "from src.data.update_training_dataset import update_training_dataset; update_training_dataset($year, $month, './data/preprocessed/')"

        echo "Splitting dataset for $year-$month"
        python -c "import pandas as pd; from src.data.split_dataset import split_data; df = pd.read_csv('./data/preprocessed/training_data.csv', index_col=False); split_data(df, './data/preprocessed/')"

        echo "Analyzing drift for $year-$month"
        python -c "from src.models.data_drift import main as drift; drift('data/preprocessed', $month, $year)"

        echo "Training model for $year-$month"
        python -c "from src.models.train_random_forests import train_rfc; train_rfc()"

        dvc add data/preprocessed/training_data.csv metrics/RandomForests_scores.json models/best_random_forests.joblib
        git add data/preprocessed/training_data.csv.dvc metrics/RandomForests_scores.json.dvc models/best_random_forests.joblib.dvc
        git commit -m "update the dvc data tables for $year-$month"
        git push origin HEAD
        dvc push -r origin data/preprocessed/training_data.csv metrics/RandomForests_scores.json models/best_random_forests.joblib
        echo "Tracked updated files with DVC"
    done
done



