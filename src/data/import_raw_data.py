import requests
import os
import logging
from src.data.check_structure import check_existing_file, check_existing_folder


def import_raw_data(raw_data_relative_path, 
                    filenames,
                    download_link):
    '''import filenames from bucket_folder_url in raw_data_relative_path'''
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    # download all the files
    for filename, identifier in filenames.items():
        input_file = os.path.join(download_link, identifier)
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
                
def main(raw_data_relative_path="./data/raw", 
        filenames = {"caracteristiques-2023.csv" : "104dbb32-704f-4e99-a71e-43563cb604f2",
                     "lieux-2023.csv": "8bef19bf-a5e4-46b3-b5f9-a145da4686bc",
                     "usagers-2023.csv": "68848e2a-28dd-4efc-9d5f-d512f7dbe66f", 
                     "vehicules-2023.csv": "146a42f5-19f0-4b3e-a887-5cd8fbef057b"},
        download_link= "https://www.data.gouv.fr/fr/datasets/r/"          
        ):
    """ Download data from french governmental platform and save it in the raw folder 
    """
    import_raw_data(raw_data_relative_path, filenames, download_link)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_fmt)
    
    main()
