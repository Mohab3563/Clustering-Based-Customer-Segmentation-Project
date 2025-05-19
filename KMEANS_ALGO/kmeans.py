import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Step 1: Load the preprocessed data
df_scaled = pd.read_csv("C:/Users/LENOVO/Desktop/subjects/CI/project/KMEANS_ALGO/mall_customers_preprocessed2.csv")
X = df_scaled.values  # Convert to numpy array for calculations

# Step 2: Define K-Means functions from scratch
def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], size=k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def has_converged(old_centroids, new_centroids, tol=1e-4):
    return np.linalg.norm(new_centroids - old_centroids) < tol

def k_means(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if has_converged(centroids, new_centroids):
            print(f"Converged after {i+1} iterations")
            break
        centroids = new_centroids
    return labels, centroids

# Step 3: Elbow Method for Optimal k
def elbow_method(X, max_k=10):
    distortions = []
    for k in range(2, max_k + 1):
        labels, centroids = k_means(X, k)
        distortion = np.sum((X - centroids[labels]) ** 2)
        distortions.append(distortion)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), distortions, marker='o')
    plt.title("Elbow Method For Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.grid(True)
    plt.show()

# Run elbow method first to decide best k
elbow_method(X, max_k=10)

# Step 4: Run K-Means with chosen k
k = 4  # Choose based on elbow plot
labels, centroids = k_means(X, k)

# Step 5: Visualize clustering in 3D (raw features)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='Set2', s=60)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=200, marker='X', label='Centroids')
ax.set_title("K-Means Clustering (3D View on First 3 Features)")
ax.set_xlabel(df_scaled.columns[0])
ax.set_ylabel(df_scaled.columns[1])
ax.set_zlabel(df_scaled.columns[2])
ax.legend()
plt.colorbar(scatter, ax=ax)
plt.tight_layout()
plt.show()

# Step 6: Visualize clustering using PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='Set2', s=60)
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
ax.set_title("K-Means Clustering with PCA (3D Projection)")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
plt.grid(True)
plt.show()
