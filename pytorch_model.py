import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Load the ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract embedding from an image
def extract_embedding(image_path):
    """
    Extracts a 512-dimensional face embedding from an image.
    """
    try:
        # Load Image
        image = Image.open(image_path).convert("RGB")  # Convert to RGB

        # Preprocess
        preprocessed_image = transform(image).unsqueeze(0)  # Add batch dimension

        # Get Embedding
        with torch.no_grad():
            embedding = model(preprocessed_image)

        embedding = embedding.numpy().flatten()  # Convert to NumPy array

        if embedding.shape[0] != 512:
            print(f"Invalid embedding shape for {image_path}: {embedding.shape}")
            return None
        
        return embedding

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Test the function
if __name__ == "__main__":
    image_path = "dataset/2024076335_Sai Manish reddy/2.jpg"  # Replace with an actual image pathdataset/2024076335_Sai Manish reddy
    embedding = extract_embedding(image_path)

    if embedding is not None:
        print(f"Extracted embedding shape: {embedding.shape}")
    else:
        print("Failed to extract embedding.")
