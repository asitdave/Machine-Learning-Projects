# src/feature_engineering.py
# Note: Handles creation of new features.

import numpy as np

def add_features(data):
    """Add additional derived features to the dataset."""
    data['area_per_bedroom'] = data['area'] / data['bedrooms']
    data['area_per_bathroom'] = data['area'] / data['bathrooms']
    data['area_per_story'] = data['area'] / data['stories']
    data['price_per_sqft'] = data['price'] / data['area']
    data['log_price'] = np.log10(data['price'])
    return data
