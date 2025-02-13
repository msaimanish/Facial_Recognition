import cv2
import os
import time
import numpy as np
from pytorch_model import extract_embedding
from database import insert_face

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
