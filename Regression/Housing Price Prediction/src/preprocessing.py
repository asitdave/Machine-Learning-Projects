# src/preprocessing.py
# Note: Handles data cleaning, outlier removal, encoding, and splitting

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(data):
    """Remove rows with bedrooms outside the range [2, 4]."""
    data = data[(data['bedrooms'] > 1) & (data['bedrooms'] < 5)]
    return data

def remove_outliers(data):
    """Remove outliers using Tukey's method."""
    data_summary = data.describe().T
    data_summary['IQR'] = data_summary['75%'] - data_summary['25%']
    data_summary['lower_bound'] = data_summary['25%'] - 1.5 * data_summary['IQR']
    data_summary['upper_bound'] = data_summary['75%'] + 1.5 * data_summary['IQR']
    cols_without_string_elmts = data.select_dtypes(include=[np.number]).columns
    filtered_data = data[cols_without_string_elmts]
    outliers = (filtered_data < data_summary['lower_bound']) | (filtered_data > data_summary['upper_bound'])
    data = data[outliers.any(axis=1) != True]
    return data

def encode_categorical(data):
    """Encode categorical variables using LabelEncoder."""
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column])
    return data

def split_data(data):
    """Split data into train and test sets."""
    X = data[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
              'parking', 'prefarea', 'furnishingstatus', 'area_per_bedroom',
              'area_per_bathroom', 'area_per_story', 'price_per_sqft']]
    Y = data['log_price']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    return X, Y, X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Standardize the features using z-scores."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
