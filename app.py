import cv2
import torch
import numpy as np
from pytorch_model import extract_embedding
from database import get_all_faces
from datetime import datetime
import pandas as pd

# 🟢 Load face embeddings from database
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

print("✅ Loaded Embeddings, Shape:", known_embeddings.shape)

# 🟢 Start Webcam
cam = cv2.VideoCapture(0)
attendance = {}

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Error capturing frame, skipping...")
        continue

    # 🟢 Extract Embedding
    image_path = "temp.jpg"
    cv2.imwrite(image_path, frame)
    embedding = extract_embedding(image_path)

    if embedding is None or embedding.shape[0] != 512:
        print("❌ Error extracting embedding, skipping frame.")
        cv2.putText(frame, "Face Not Recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # 🟢 Compare with Stored Embeddings (Cosine Similarity)
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
            print(f"✅ Recognized: {name} (Confidence: {confidence:.2f})")
        else:
            cv2.putText(frame, "Face Not Recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            print("❌ Face not recognized.")

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

# 🟢 Save Attendance
df = pd.DataFrame.from_dict(attendance, orient="index")
df.to_csv("attendance.csv", index=False)
print("✅ Attendance saved to attendance.csv")
