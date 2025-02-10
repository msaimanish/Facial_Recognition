import pandas as pd
import tkinter as tk
from tkinter import ttk

def show_attendance():
    df = pd.read_csv("attendance.csv")
    for i, row in df.iterrows():
        tree.insert("", "end", values=(row["name"], row["time"]))

root = tk.Tk()
root.title("Attendance Records")

tree = ttk.Treeview(root, columns=("Name", "Time"), show="headings")
tree.heading("Name", text="Name")
tree.heading("Time", text="Time")
tree.pack()

show_attendance()
root.mainloop()
