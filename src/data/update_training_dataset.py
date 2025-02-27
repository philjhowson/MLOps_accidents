import os
import pandas as pd
import logging
def update_training_dataset(year, month, output_folderpath):
    simulated_df = pd.read_csv(os.path.join(output_folderpath, 'simulated_data.csv'))
    training_df = pd.read_csv(os.path.join(output_folderpath, 'training_data.csv'))

    temp_df = simulated_df[(simulated_df['mois'] == month) & (simulated_df['year_acc'] == year)]
    training_df = pd.concat([training_df, temp_df])

    training_df.to_csv(os.path.join(output_folderpath, 'training_data.csv'), index=False)
    simulated_df.to_csv(os.path.join(output_folderpath, 'simulated_data.csv'), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('update training data set with simulation data')
    update_training_dataset(2022, 1, './data/preprocessed/')