import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.data.make_dataset.py import main

def test_make_dataset():

    main()

    
