import argparse
import os
import torch

from src.data_preparation import load_data
from src.model import FMNISTClassifier
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image
from src.utils import save_model, load_model

def main(args):
    # Step 1: Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(batch_size=args.batch_size)
    
    # Step 2: Define Model
    model = FMNISTClassifier()
    
    # Step 3: Train the Model (if specified)
    if args.train:
        print("Training the model...")
        train_model(model, train_loader, val_loader, epochs=args.epochs, learning_rate=args.lr)
        os.makedirs('saved_models', exist_ok=True)
        save_model(model, 'saved_models/fmnist_model.pth')
    
    # Step 4: Load Pretrained Model (if specified)
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        load_model(model, args.load_model)
    
    # Step 5: Evaluate the Model on Test Set (if specified)
    if args.evaluate:
        print("Evaluating the model...")
        evaluate_model(model, test_loader)
    
    # Step 6: Predict on New Images (if specified)
    if args.predict_image:
        print(f"Predicting on image: {args.predict_image}...")
        class_idx = predict_image(model, args.predict_image)
        print(f"Predicted Class: {class_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification with FashionMNIST")
    
    # Arguments for controlling operations
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--load_model', type=str, help="Path to a pretrained model to load.")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the model on the test set.")
    parser.add_argument('--predict_image', type=str, help="Path to an image for prediction.")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and testing.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training.")
    
    args = parser.parse_args()
    main(args)