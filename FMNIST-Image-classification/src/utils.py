import torch
import matplotlib.pyplot as plt

def save_model(model, path):
    """
    Save the model's state dictionary to the given path.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    Load the model's state dictionary from the given path.
    """
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")

def plot_metrics(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    """
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()