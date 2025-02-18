class Config:
    DATA_DIR = 'data/'
    NEW_DATA_DIR = 'data/new_data/'             # Place for saving fresh new data, which can be used for retraining
    NEW_DATA_BACKUP_DIR = 'data/new_data/old/'
    RAW_DATA_DIR = 'data/raw/'                  # Place for saving validated dats in the early stage of training pipeline
    PROCESSED_DATA_DIR = 'data/preprocessed/'      # Place for saving preprocessed and feature engineered data
    TRAINED_MODEL_DIR = 'models/'       # Place for saving trained models
    PREDICTIONS_DATA_DIR = 'data/predictions/'  # Place for saving prediction results
    DATA_DRIFT_MONOTOR_DIR = 'data_drift/' # place for saving data monitoring results

    NEW_DATA_FILE = 'data/new_data/new_data.csv'    # contains fresh new data
    OUTPUT_TRAINED_MODEL_FILE_RF = 'models/model_best_rf'    # Trained random forest classifer model file. We will skip joblib extension
    OUTPUT_TRAINED_MODEL_FILE_RF_DISCARDED = 'models/discarded/model_best_rf'    # Trained random forest classifer file which has not a good accuracy. We will skip joblib extension
    OUTPUT_PREDICTIONS_RESULTS_FILE = 'data/predictions/predictions.csv'  # Prediction results

    BUCKET_FOLDER_URL = "https://mlops-project-db.s3.eu-west-1.amazonaws.com/accidents/"
    