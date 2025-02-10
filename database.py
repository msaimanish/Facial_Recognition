import sqlite3
import numpy as np

# 🟢 Connect & Create Table
def init_db():
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database and table initialized!")

# 🟢 Insert or Update Face Data
def insert_face(student_id, name, embedding):
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    embedding_bytes = embedding.tobytes()

    try:
        cursor.execute("""
            INSERT INTO faces (student_id, name, embedding) 
            VALUES (?, ?, ?)
            ON CONFLICT(student_id) 
            DO UPDATE SET name=excluded.name, embedding=excluded.embedding
        """, (student_id, name, embedding_bytes))

        conn.commit()
        print(f"✅ Face data saved/updated for {name} (ID: {student_id})")

    except sqlite3.Error as e:
        print(f"❌ Database Error: {e}")

    finally:
        conn.close()

# 🟢 Retrieve All Faces
def get_all_faces():
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("SELECT student_id, name, embedding FROM faces")
    faces = cursor.fetchall()
    conn.close()
    return faces

# 🟢 Initialize Database
if __name__ == "__main__":
    init_db()
    faces = get_all_faces()
    print(f"🔍 Total Records in DB: {len(faces)}")
