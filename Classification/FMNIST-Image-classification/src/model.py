import torch.nn as nn
import torch.nn.functional as F

class FMNISTClassifier(nn.Module):
    """
    Neural Network Model for FMNIST Classification.
    """
    def __init__(self):
        super(FMNISTClassifier, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Convolutional Layer 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Convolutional Layer 2
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)  # Fully Connected Layer 1 (Connects all input neurons to all output neurons)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Fully Connected Layer 2 (Output)

        self.pool = nn.MaxPool2d(2, 2)  # Pooling Layer (Reduce the spatial dimensions to focus on the most important elements)
        self.dropout = nn.Dropout(0.25)  # Dropout to prevent overfitting

    def forward(self, x):
        # Convolution + Pooling + Activation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the data for Fully Connected Layers
        x = x.view(-1, 64 * 7 * 7)

        # Fully Connected Layers + Activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer (logits)

        return x

if __name__ == "__main__":
    model = FMNISTClassifier()
    print(model)