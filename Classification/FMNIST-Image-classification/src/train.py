import torch
import torch.nn as nn
import torch.optim as optim
from src.data_preparation import load_data
from src.model import FMNISTClassifier

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """
    Train the model and validate it at each epoch.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(f"Test Accuracy: {correct/total:.4f}")

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {correct/total:.4f}")
        
    
    print("Training complete.")
    return model

if __name__ == "__main__":
    train_loader, val_loader, _ = load_data()
    model = FMNISTClassifier()
    trained_model = train_model(model, train_loader, val_loader)
    torch.save(trained_model.state_dict(), "saved_models/fmnist_model.pth")