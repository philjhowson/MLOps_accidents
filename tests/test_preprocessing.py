import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src/data/')))

import pytest
import pandas as pd
from make_dataset import main

def test_make_dataset():

    main()

    path1 = 'data/preprocessed'
    path2 = 'data/preprocessed/X_train.csv'
    path3 = 'data/preprocessed/X_test.csv'
    path4 = 'data/preprocessed/y_train.csv'
    path5 = 'data/preprocessed/y_test.csv'

    assert os.path.exists(path1), f"Folder @ {path1} not found."
    assert os.path.exists(path2), f"File @ {path2} not found."
    assert os.path.exists(path3), f"File @ {path3} not found."
    assert os.path.exists(path4), f"File @ {path4} not found."
    assert os.path.exists(path5), f"File @ {path5} not found."

    X_train = pd.read_csv(path2)
    X_test = pd.read_csv(path3)
    y_train = pd.read_csv(path4)
    y_test = pd.read_csv(path5)
        
    assert len(X_train) == len(y_train), f"X_train and y_train have mismatched dimensions."
    assert len(X_test) == len(y_test), f"X_test and y_test have mismatched dimensions."
    assert max(y_train['grav']) == 1, f"y_train has incorrect encoding."
    assert max(y_test['grav']) == 1, f"y_test has incorrect encoding."
