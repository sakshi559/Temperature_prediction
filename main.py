import torch
import cv2
import numpy as np
from torchvision.models import resnet18
from torchvision.transforms import transforms

# Create a new instance of the ResNet model
model = resnet18(pretrained=True)
num_features = model.fc.in_features
# Output layer for temperature prediction
model.fc = torch.nn.Linear(num_features, 1)

# Load the state dictionary from the saved model
state_dict = torch.load('model.pth')

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Load and preprocess the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = ['./Images/infrared/image3.jpg'] #change this
images = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    images.append(image)

# Stack the images into a single tensor
image_tensor = torch.stack(images)

# Make the prediction
with torch.no_grad():
    prediction = model(image_tensor)

# Convert the prediction to a numpy array
prediction = prediction.numpy()

warning_temp = 30
# Print the predicted temperatures
for i, image_path in enumerate(image_paths):
  print(f"Predicted temperature for {image_path}: {prediction[i][0]}")
  if prediction[i][0] > warning_temp:
      print("WARNING: Temperature is too high!")
