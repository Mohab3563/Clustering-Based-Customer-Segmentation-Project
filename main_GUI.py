import tkinter as tk
import subprocess
import os
import sys

def open_fcm_gui():
    path = os.path.join("FCM_ALGO", "gui.py")
    subprocess.Popen([sys.executable, path])

def open_kmeans_gui():
    path = os.path.join("KMEANS_ALGO", "gui.py")
    subprocess.Popen([sys.executable, path])

root = tk.Tk()
root.title("Clustering Algorithm Launcher")

tk.Label(root, text="Select a Clustering Method", font=("Helvetica", 16)).pack(pady=20)

tk.Button(root, text="Fuzzy C-Means GUI", command=open_fcm_gui, width=30, height=2).pack(pady=10)
tk.Button(root, text="K-Means GUI", command=open_kmeans_gui, width=30, height=2).pack(pady=10)

root.mainloop()
