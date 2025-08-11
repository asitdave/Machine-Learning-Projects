import torch
from torchvision import transforms
from PIL import Image
from src.model import FMNISTClassifier

def predict_image(model, image_path):
    """
    Predict the class of a single image.
    """
    # Define transformation for the input image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

if __name__ == "__main__":
    model = FMNISTClassifier()
    model.load_state_dict(torch.load('saved_models/fmnist_model.pth'))
    class_idx = predict_image(model, 'path_to_your_image.jpg')
    print(f"Predicted Class: {class_idx}")