# src/models.py
# Note: Handles model initialization, training, and evaluation.

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import numpy as np
import pandas as pd

def train_linear_regression(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=8):
    """Train a decision tree model."""
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=40, max_depth=25):
    """Train a random forest model."""
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using MAPE and R2 score."""
    y_pred = model.predict(X_test)
    mse = metrics.mean_absolute_percentage_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    return mse, r2

def feature_importance(model, model_name):
    """Get feature importance from a model."""
    print(model.__class__.__name__)
    if model_name == "Linear Regression":
        return np.abs(model.coef_)
    return model.feature_importances_

def get_feature_importance_df(models, filtered_data):
    """Get feature importance DataFrame for multiple models."""
    feature_names = filtered_data.columns if hasattr(filtered_data, "columns") else [f"Feature {i}" for i in range(filtered_data.shape[1])]
    importance_df = pd.DataFrame({"Feature": feature_names})
    for model_name, model in models.items():
        importance = feature_importance(model, model_name)
        importance_df[model_name] = importance

    # Melt the DataFrame for easier plotting with seaborn
    importance_melted = importance_df.melt(id_vars="Feature", var_name="Model", value_name="Importance")
    return importance_melted

