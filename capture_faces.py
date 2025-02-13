import cv2
import os
import time
import numpy as np
from database import insert_face
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

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

# User input for student details
name = input("Enter Your Name: ").strip()
student_id = input("Enter Your Student ID: ").strip()

# Create folder for student
folder_path = os.path.join("dataset", f"{student_id}_{name}")
os.makedirs(folder_path, exist_ok=True)

# Start webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Capturing 10 images for {name}...")

captured_images = []
for i in range(10):
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
        continue

    # Save image
    img_path = os.path.join(folder_path, f"{i}.jpg")
    cv2.imwrite(img_path, frame)
    captured_images.append(img_path)

    # Display frame with countdown
    cv2.putText(frame, f"Capturing {i+1}/10...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Capture Face", frame)

    print(f"Saved image: {img_path}")

    time.sleep(0.5)  # Wait 0.5 seconds before capturing the next image

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

# Extract embeddings from images
print("Extracting embeddings...")
valid_embeddings = []
for img_path in captured_images:
    embedding = extract_embedding(img_path)
    if embedding is not None and embedding.shape[0] == 512:
        valid_embeddings.append(embedding)

# Store face data in the database
if valid_embeddings:
    avg_embedding = np.mean(valid_embeddings, axis=0)  # Average embedding
    insert_face(student_id, name, avg_embedding)
    print(f"Face data saved for {name} (ID: {student_id})")
else:
    print("Failed to extract valid embeddings.")
