
# Image Classification Using Neural Networks (FashionMNIST)

This project demonstrates the creation, training, and evaluation of a deep learning model to classify images from the FashionMNIST dataset into 10 categories, such as T-shirts, Trousers, Pullovers, etc. The goal is to use a neural network to accurately predict the correct category for each image and test the model on unseen data, including real-world images.


### Project Highlights

- **Objective**: To build a neural network for image classification and evaluate its performance on the FashionMNIST dataset.

- **Key Features**:
	-   Data exploration and preprocessing to understand the dataset and prepare it for training.
	-	Implementation of a custom neural network with PyTorch.
	-	Model training and validation with real-time accuracy and loss tracking.
	-	Evaluation using metrics like accuracy and a confusion matrix.

- **Tools & Libraries**:
	-	Pandas, NumPy, Matplotlib, Seaborn, scikit-learn

### Results

The neural network achieves:
- Validation Accuracy: ~98% after training for 10 epochs
- Test Accuracy: ~90%

The performance of the model is visualized using:
- Training/validation loss and accuracy plots
- A confusion matrix to identify misclassified classes

You can also test the model on custom real-world images using the scripts provided.


### Real-World Image Testing

To test the model on external images, place your image files in a designated directory (e.g., test_images/) and modify the paths in `evaluate.py` accordingly. Use `main.py` or the notebook to predict their labels.

[Main page](/)