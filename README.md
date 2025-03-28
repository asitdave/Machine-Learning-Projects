
# Machine Learning Projects

Welcome to my machine learning projects repository! This portfolio contains a series of hands-on projects designed to demonstrate various machine learning techniques, algorithms, and workflows. Each project explores different aspects of data science, from data cleaning and preprocessing to model training and evaluation.


## Recent Project

[User Segmentation and App Uninstall Prediction](./Clustering/App-Users-Segmentation/)

**Objective**:
- Identify key factors influencing user retention and churn.
- Segment users based on behavior using clustering techniques.
- Train machine learning models to predict app uninstallation.
- Implement an interactive system for real-time user predictions.

**Techniques Used**:
- User Segmentation: Clustering with K-Means to categorize user behaviors.
- Churn Prediction: Classification using Random Forest and XGBoost.

## Future Projects

This repository will continue to grow as I add more machine learning projects. Here are the general areas that will be covered:

-   Supervised Learning: Classification and Regression problems
-	Unsupervised Learning: Clustering and Dimensionality Reduction
-	Deep Learning: Neural Networks and Computer Vision

## List of Projects

| Sr. No. | Project Name    | Category    | Short Description        |
|:-------:|:----------------|:-----------:|:-------------------------|
| 1       | [Housing Price Prediction](./Regression/Housing%20Price%20Prediction/)   | Regression   | Develops a regression model to predict house prices based on various features such as amenities, area, etc. |
| 2       | [FMNIST Image Classification](./Classification/FMNIST-Image-classification/)    | Classification      | Implements a neural network to classify Fashion-MNIST images into categories like shirts, shoes, and bags.   |
| 3       | [Transaction Anomaly Detection](./Classification/Transaction-Anomaly-Detection/)   | Classification  | Detecting anomalies in financial transactions using Isolation Forest.      |
| 4       | [User Segmentation and App Uninstall Prediction](./Clustering/App-Users-Segmentation/)   | Clustering     | Analyze user behaviour to segment users and predict app uninstallation using K-means, Random Forest and XGBoost.   |


## Folder Structure

Here's the organization of the repository:

```
/Machine-Learning-Projects
│
├── /Regression
│   └── /Housing Price Prediction
│
├── /Classification
│   ├── /FMNIST-Image-Classification
│   └── /Transaction-Anomaly-Detection
│	
├── /Clustering
│	└── /App-Users-Segmentation
│
└── README.md
```

<!-- /FMNIST-Image-Classification
│       ├── saved_models
│		│	└── fmnist_model.pth
│		├──src
│	    │   ├── __init__.py
│		│	├── data_preparation.py
│		│	├── model.py
│		│	├── train.py
│		│	├── predict.py
│	    │   ├── evaluate.py
│       │   ├── utils.py
│       ├── main.py
│       ├── NN-Classification-FMNIST.ipynb
│       └── README.md -->



## How to Run

To run the code in this repository, perform the following steps:

1. Clone the repository
```bash
git clone https://github.com/asitdave/Machine-Learning-Projects.git
```

2. Create a virtual environment to install required dependencies.

```bash
pip install -r requirements.txt
```

or (for Anaconda)

```bash
conda env create -f ML-proj-venv.yml
conda activate ML-proj-venv.yml
```



## Contributing

Feel free to fork this repository, explore the code, and contribute by submitting issues or pull requests. Suggestions are always welcome!
