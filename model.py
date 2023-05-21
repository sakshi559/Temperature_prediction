import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Define the hyperparameters
image_size = 224
channels = 3
num_epochs = 10
batch_size = 28
learning_rate = 0.001

# Load the pre-trained ResNet model
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
# Output layer for temperature prediction
model.fc = nn.Linear(num_features, 1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Set up the data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Load and preprocess the dataset
dataset = torchvision.datasets.ImageFolder(root='Images', transform=transform)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

# Define the list of temperatures globally
train_temperatures = [
  106.2, 80.6, 34.7, 37.3, 61.2, 
  52.6, 58.8, 58.1, 37.3, 58.8,
  65.3, 98.2, 36.7, 37.8, 106.7, 
  37.2, 40.5, 105, 27.5, 25.3,
  41.5, 51.6, 34.7, 71.9, 40.8, 
  39.0, 39.1, 76.0
] # in Celsius


# Training loop
for epoch in range(num_epochs):
  for images, _ in data_loader:
    # Convert input data to Float type
    images = images.to(torch.float32)

    # Get temperatures for the current batch
    temperatures = train_temperatures[:len(images)]
    temperatures = torch.tensor(temperatures).to(torch.float32)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, temperatures.unsqueeze(1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # Print the loss after each epoch
  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
