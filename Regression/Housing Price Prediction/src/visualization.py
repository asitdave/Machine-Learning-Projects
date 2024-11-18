# src/visualization.py
# Note: Handles data visualization and plotting

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_price_distribution(data):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, x='price', hue='bedrooms', kde=False, bins=30, 
                 palette='inferno', multiple='stack', shrink=0.8,
                 line_kws={'linewidth': 1, 'ls':'--'})
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Counts')
    plt.show()

def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="crest", linewidths=.5, linecolor='white')
    plt.title('Correlation Matrix')
    plt.show()

def plot_model_performance(y_test, y_preds, model_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, ax in enumerate(axes):
        sns.scatterplot(x=y_test, y=y_preds[i], ax=ax, color="blue", alpha=0.6, label="Predictions")
        sns.regplot(x=y_test, y=y_preds[i], ax=ax, scatter=False, color="red")
        ax.set_title(model_names[i])
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
    plt.show()

def plot_feature_importance(importance_df, model_names):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Model", palette="viridis")
    plt.title(f"Feature Importances Comparison: {', '.join(model_names)}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.legend(title="Model")
    plt.show()