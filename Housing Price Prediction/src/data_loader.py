# src/data_loader.py
# Note: Handles loading and initial exploration of the dataset

import pandas as pd

def load_data(file_path):
    """Load dataset from a given file path."""
    data = pd.read_csv(file_path)
    return data

def explore_data(data):
    """Perform initial exploration of the dataset."""
    print(data.info())
    print(data.head())
    return data
