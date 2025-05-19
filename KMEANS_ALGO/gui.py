import tkinter as tk
from tkinter import messagebox
from kmeans import X, df_scaled, k_means  # Use everything from your original kmeans.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensure this is present even if not directly used
import numpy as np

def run_kmeans():
    try:
        k = int(entry_k.get())
        if k < 1:
            raise ValueError("k must be >= 1")
    except ValueError as e:
        messagebox.showerror("Invalid Input", f"Please enter a valid integer for k.\n{e}")
        return

    labels, centroids = k_means(X, k)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='Set2', s=60)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=200, marker='X', label='Centroids')
    ax.set_title(f"K-Means Clustering (k={k}) - 3D View")
    ax.set_xlabel(df_scaled.columns[0])
    ax.set_ylabel(df_scaled.columns[1])
    ax.set_zlabel(df_scaled.columns[2])
    ax.legend()
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    plt.show()

# Build GUI
root = tk.Tk()
root.title("K-Means Clustering GUI")

tk.Label(root, text="Enter number of clusters (k):").pack(pady=5)
entry_k = tk.Entry(root)
entry_k.pack(pady=5)

run_button = tk.Button(root, text="Run K-Means", command=run_kmeans)
run_button.pack(pady=10)

root.mainloop()
