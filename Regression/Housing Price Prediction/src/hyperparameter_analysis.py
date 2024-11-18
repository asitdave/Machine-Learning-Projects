# src/hyperparameter_analysis.py

from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def test_dtr_hyperparameters(X_train, y_train, X_test, y_test, max_depths):
    """Test different hyperparameters for a decision tree regressor."""
    results = []
    for depth in max_depths:
        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = metrics.mean_absolute_percentage_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        results.append((depth, mse, r2))
    return results

def test_rfr_hyperparameters(X_train, y_train, X_test, y_test, n_estimators, max_depths):
    """Test different hyperparameters for a random forest regressor."""
    results = []
    for depth in max_depths:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = metrics.mean_absolute_percentage_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        results.append((depth, mse, r2))
    return results
