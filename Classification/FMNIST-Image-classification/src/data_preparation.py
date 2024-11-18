import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data(batch_size=64, val_split=0.1):
    """
    Load and preprocess the FMNIST dataset, splitting into train, validation, and test sets.
    """
    # Define transformations: Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0.5, std 0.5
    ])

    # Download and load datasets
    train_dataset = datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./datasets', train=False, download=True, transform=transform)

    # Split train dataset into train and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()
    print("Data prepared and DataLoaders created.")