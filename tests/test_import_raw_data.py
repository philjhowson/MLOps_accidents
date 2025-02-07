import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src/data/')))

import pytest
from import_raw_data import main

def test_import_data():

    main()

    file1 = 'data/raw/caracteristiques-2023.csv'
    file2 = 'data/raw/lieux-2023.csv'
    file3 = 'data/raw/usagers-2023.csv'
    file4 = 'data/raw/vehicules-2023.csv'

    assert os.path.exists(file1), f"File @ {file1} not found!"
    assert os.path.exists(file2), f"File @ {file2} not found!"
    assert os.path.exists(file3), f"File @ {file3} not found!"
    assert os.path.exists(file4), f"File @ {file4} not found!"
