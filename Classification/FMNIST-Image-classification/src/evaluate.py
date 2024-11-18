import torch
from src.data_preparation import load_data
from src.model import FMNISTClassifier

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    _, _, test_loader = load_data()
    model = FMNISTClassifier()
    model.load_state_dict(torch.load('saved_models/fmnist_model.pth'))  # Load trained model
    evaluate_model(model, test_loader)