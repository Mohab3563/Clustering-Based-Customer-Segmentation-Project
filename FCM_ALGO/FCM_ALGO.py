import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data (fixed full path)
data = pd.read_csv('C:/Users/LENOVO/Desktop/subjects/CI/project/KMEANS_ALGO/mall_customers_preprocessed2.csv')
X = data.values
n_samples, n_features = X.shape

# ------------------ FCM Core Functions ------------------

def calculate_centroids(U, X, m):
    um = U ** m
    centroids = um.T @ X / np.sum(um.T, axis=1)[:, None]
    return centroids

def update_memberships(X, centroids, m, distance_metric='euclidean'):
    if distance_metric == 'mahalanobis':
        VI = np.linalg.inv(np.cov(X.T))
        distance_matrix = cdist(X, centroids, metric='mahalanobis', VI=VI)
    elif distance_metric == 'cosine':
        distance_matrix = cdist(X, centroids, metric='cosine')
    else:
        distance_matrix = cdist(X, centroids, metric='euclidean')
    distance_matrix = np.fmax(distance_matrix, np.finfo(np.float64).eps)
    inv_distances = 1.0 / distance_matrix
    power = 2.0 / (m - 1)
    denominator = np.sum((inv_distances[:, :, None] / inv_distances[:, None, :]) ** power, axis=2)
    U_new = 1.0 / denominator
    return U_new

def objective_function(U, centroids, X, m, distance_metric='euclidean'):
    if distance_metric == 'mahalanobis':
        VI = np.linalg.inv(np.cov(X.T))
        distance_matrix = cdist(X, centroids, metric='mahalanobis', VI=VI) ** 2
    elif distance_metric == 'cosine':
        distance_matrix = cdist(X, centroids, metric='cosine') ** 2
    else:
        distance_matrix = cdist(X, centroids, metric='euclidean') ** 2
    obj = np.sum((U ** m) * distance_matrix)
    return obj

def preserve_diversity(centroids, min_dist=1e-3, perturb_scale=0.05, strategy='gaussian'):
    K = centroids.shape[0]
    for i in range(K):
        for j in range(i + 1, K):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            if distance < min_dist:
                if strategy == 'reinitialize':
                    centroids[j] = np.random.rand(centroids.shape[1])
                else:
                    centroids[j] += np.random.normal(scale=perturb_scale, size=centroids.shape[1])
    return centroids

def initialize_U(n_samples, K, method='dirichlet'):
    if method == 'uniform':
        return np.full((n_samples, K), 1.0 / K)
    else:
        return np.random.dirichlet(np.ones(K), size=n_samples)

def run_fcm(X, K, m, max_iter=150, epsilon=1e-5, random_state=None,
            init_method='dirichlet', diversity_strategy='gaussian', distance_metric='euclidean'):
    np.random.seed(random_state)
    U = initialize_U(n_samples, K, init_method)
    for _ in range(max_iter):
        centroids = calculate_centroids(U, X, m)
        centroids = preserve_diversity(centroids, strategy=diversity_strategy)
        U_new = update_memberships(X, centroids, m, distance_metric)
        if np.linalg.norm(U_new - U) < epsilon:
            break
        U = U_new
    obj = objective_function(U, centroids, X, m, distance_metric)
    return obj, centroids, U

# ------------------ Experiment Runner ------------------

def run_experiments(K=3, m=2.0, n_runs=30):
    init_methods = ['dirichlet', 'uniform']
    diversity_strategies = ['gaussian', 'reinitialize']
    distance_metrics = ['euclidean', 'mahalanobis', 'cosine']

    summary = []
    seeds_log = []

    best_obj = float('inf')
    best_config = None
    best_seed = None

    for init in init_methods:
        for diversity in diversity_strategies:
            for dist in distance_metrics:
                objs = []
                for _ in range(n_runs):
                    seed = np.random.randint(0, 1e6)
                    obj, _, _ = run_fcm(
                        X, K, m,
                        random_state=seed,
                        init_method=init,
                        diversity_strategy=diversity,
                        distance_metric=dist
                    )
                    objs.append(obj)
                    seeds_log.append(f"{init},{diversity},{dist},{seed}")
                    if obj < best_obj:
                        best_obj = obj
                        best_config = (init, diversity, dist)
                        best_seed = seed
                summary.append({
                    'Initialization': init,
                    'Diversity': diversity,
                    'Distance': dist,
                    'Average Objective': np.mean(objs),
                    'Std Objective': np.std(objs),
                    'Min Objective': np.min(objs)
                })

    # Save results 
    results_dir = "C:/Users/LENOVO/Desktop/subjects/CI/project/FCM_ALGO"
    pd.DataFrame(summary).to_csv(os.path.join(results_dir, 'FCM_results.csv'), index=False)
    with open(os.path.join(results_dir, 'FCM_seeds.txt'), 'w') as f:
        f.write('\n'.join(seeds_log))

    print(f" Summary saved to {results_dir}/FCM_results.csv")
    print(f" Seeds saved to {results_dir}/FCM_seeds.txt")
    print(" Summary saved to FCM_results.csv")
    print(" Seeds saved to FCM_seeds.txt")

    return best_config, best_seed, best_obj

# ------------------ Plot Best Configuration ------------------

def plot_best_clusters(X, U, centroids, obj, seed, config):
    labels = np.argmax(U, axis=1)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(centroids.shape[0]):
        ax.scatter(X[labels == k, 0], X[labels == k, 1], X[labels == k, 2], label=f'Cluster {k+1}')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='black', s=100, marker='x', label='Centroids')

    ax.set_title(
        f"Best FCM Clustering\n"
        f"Obj: {obj:.2f} | Seed: {seed}\n"
        f"Init: {config[0]} | Diversity: {config[1]} | Distance: {config[2]}"
    )

    ax.legend()
    plt.tight_layout()
    plt.show()

# ------------------ Run Everything ------------------

if __name__ == '__main__':
    K = 4
    m = 2.0
    best_config, best_seed, best_obj = run_experiments(K=K, m=m, n_runs=30)

    print(f"\n Best Configuration: {best_config} | Seed: {best_seed} | Objective: {best_obj:.4f}\n")

    # Run best again to get centroids and membership for plotting
    obj, centroids, U = run_fcm(
        X, K, m,
        random_state=best_seed,
        init_method=best_config[0],
        diversity_strategy=best_config[1],
        distance_metric=best_config[2]
    )
    plot_best_clusters(X, U, centroids, obj, best_seed, best_config)



def elbow_method_fcm(X, m=2.0, k_range=range(2, 11), n_runs=5,
                     init_method='dirichlet', diversity_strategy='gaussian', distance_metric='euclidean'):
    avg_objectives = []

    for K in k_range:
        objs = []
        print(f"Evaluating K = {K}...")
        for _ in range(n_runs):
            seed = np.random.randint(0, 1e6)
            obj, _, _ = run_fcm(
                X, K, m,
                random_state=seed,
                init_method=init_method,
                diversity_strategy=diversity_strategy,
                distance_metric=distance_metric
            )
            objs.append(obj)
        avg_obj = np.mean(objs)
        avg_objectives.append(avg_obj)

    # Plot Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), avg_objectives, marker='o', linestyle='--', color='blue')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Average Objective Function")
    plt.title("Elbow Method for Optimal K (FCM)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Optional: Tune K using Elbow Method
elbow_method_fcm(
    X, m=2.0, k_range=range(2, 11),
    n_runs=5,
    init_method='dirichlet',
    diversity_strategy='gaussian',
    distance_metric='euclidean'
)
