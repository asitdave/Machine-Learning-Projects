
# Machine Learning Projects

Welcome to my machine learning projects repository! This portfolio contains a series of hands-on projects designed to demonstrate various machine learning techniques, algorithms, and workflows. Each project explores different aspects of data science, from data cleaning and preprocessing to model training and evaluation.


## Recent Project

[Classification: Image Classification using Neural Networks (FasionMNIST)](./Classification/FMNIST-Image-classification/)


- **Objective**: Build a neural network to classify images from the FashionMNIST dataset into 10 categories (e.g., T-shirt, Trouser, Dress).

- **Techniques**:

	-   Data exploration and processing
	-	Neural network design (Convolutional Neural Networks - CNNs)
	-	Model training, validation, and testing
	-	Evaluation using metrics like accuracy and confusion matrix

- **Tools & Libraries**:
	-	PyTorch, NumPy, Matplotlib, Seaborn

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
<!-- | 3       | Project Gamma   | Technology  | AI-powered chatbot.      |
| 4       | Project Delta   | Finance     | Personal finance tool.   | -->


## Folder Structure

Here's the organization of the repository:

```
/Machine-Learning-Projects
│
├── /Regression
│   └── /Housing Price Prediction
│
├── /Classification
│   └── /FMNIST-Image-Classification
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
│       └── README.md
│
└── README.md
```
    
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
