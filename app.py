import cv2
import numpy as np
from database import get_all_faces
from datetime import datetime
import pandas as pd
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

# Load face embeddings from database
faces = get_all_faces()

known_embeddings = []
known_ids = []
known_names = []

for face in faces:
    embedding = np.frombuffer(face[2], dtype=np.float32)  # Convert bytes to array
    if embedding.shape[0] == 512:  # Ensure correct shape
        known_embeddings.append(embedding)
        known_ids.append(face[0])
        known_names.append(face[1])

if not known_embeddings:
    print("⚠️ No face data found in the database!")
    exit()

known_embeddings = np.array(known_embeddings)

print("Loaded Embeddings, Shape:", known_embeddings.shape)

# Start Webcam
cam = cv2.VideoCapture(0)
attendance = {}

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error capturing frame, skipping...")
        continue

    # Extract Embedding
    image_path = "temp.jpg"
    cv2.imwrite(image_path, frame)
    embedding = extract_embedding(image_path)

    if embedding is None or embedding.shape[0] != 512:
        print("Error extracting embedding, skipping frame.")
        cv2.putText(frame, "Face Not Recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Compare with Stored Embeddings (Cosine Similarity)
        similarities = np.dot(known_embeddings, embedding) / (
            np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding)
        )

        if len(similarities) == 0:
            print("⚠️ No known faces in the database!")
            continue

        match_index = np.argmax(similarities)
        confidence = similarities[match_index]

        if confidence > 0.75:  # Threshold for recognition
            student_id = known_ids[match_index]
            name = known_names[match_index]

            if student_id not in attendance:
                attendance[student_id] = {
                    "name": name,
                    "time": datetime.now().strftime("%H:%M:%S")
                }

            cv2.putText(frame, f"{name} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Recognized: {name} (Confidence: {confidence:.2f})")
        else:
            cv2.putText(frame, "Face Not Recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            print("Face not recognized.")

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

# Save Attendance
df = pd.DataFrame.from_dict(attendance, orient="index")
df.to_csv("attendance.csv", index=False)
print("Attendance saved to attendance.csv")
