# main.py

# Note: Coordinates the entire workflow from loading to model training and evaluation.

from src.data_loader import load_data, explore_data
from src.preprocessing import clean_data, remove_outliers, encode_categorical, split_data, scale_features
from src.feature_engineering import add_features
from src.models import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model, get_feature_importance_df
from src.visualization import plot_price_distribution, plot_correlation_matrix, plot_model_performance, plot_feature_importance

import joblib

# Load and explore the data
data = load_data("data/data.txt")
explore_data(data)

# Preprocess the data
data = clean_data(data)
data = remove_outliers(data)
data = encode_categorical(data)
data = add_features(data)

# Visualize data
plot_price_distribution(data)
plot_correlation_matrix(data)

# Split and scale data
X, Y, X_train, X_test, y_train, y_test = split_data(data)
X_train, X_test = scale_features(X_train, X_test)

# Train and evaluate models
models = {
    "Linear Regression": train_linear_regression(X_train, y_train),
    "Decision Tree": train_decision_tree(X_train, y_train),
    "Random Forest": train_random_forest(X_train, y_train)
}

# Plot performance
y_preds = [model.predict(X_test) for model in models.values()]
plot_model_performance(y_test, y_preds, list(models.keys()))

# Feature importance
f_imps_df = get_feature_importance_df(models, X)
plot_feature_importance(f_imps_df, list(models.keys()))

# Evaluate models
results = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}
print(results)

# Save the best model, which has lowest MAPE and highest R2 score
best_model = min(results, key=lambda x: results[x][0])
print("\n------------------------")
print(f"Best model: {best_model}")
print(f"MAPE: {results[best_model][0]}")
print(f"R2 Score: {results[best_model][1]}")

joblib.dump(best_model, "models/best_model.pkl")

