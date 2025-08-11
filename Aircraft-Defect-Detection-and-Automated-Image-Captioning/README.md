## Aircraft Defect Detection and Automated Image Captioning

This project develops a deep learning pipeline to detect defects in aircraft images and automatically generate descriptive captions summarizing the findings. The solution combines image classification for defect detection with image captioning using pretrained models, enabling both visual recognition and automated reporting for real-world inspection scenarios.


### Project highlights
- **Objective**:
	- Use the VGG16 pretrained model for image classification.
	- Prepare and preprocess image data.
	- Evaluate the model's performance and visualize predictions on test data.
	- Use BLIP pretrained model for image captioning and summarization.
	- Generate caption and summary for an aircraft image.

- **Techniques Used**:
	- **Image Preprocessing**: Rescaling and preparing datasets with Keras ImageDataGenerator.
	- **Transfer Learning**: Using VGG16 (trained on ImageNet) for feature extraction.
	- **Defect Detection**: Image Classification using Convolutional Neural Networks (CNNs) and transfer learning.
	- **Image Captioning**: Transformer-based BLIP model to describe and summarize defects
	- **Custom Keras Layer**: Integrated BLIP within a custom TensorFlow/Keras layer for task-specific text generation.

- **Tools & Libraries**:
	- **Deep Learning**: TensorFlow, Keras
	- **Pre-trained Models**: VGG16 and BLIP
	- **Data Visualization**: Matplotlib


[Main page](/)