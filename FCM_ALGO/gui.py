import tkinter as tk
from tkinter import ttk, messagebox
from FCM_ALGO import run_fcm, plot_best_clusters, X

class FCMApp:
    def __init__(self, master):
        self.master = master
        master.title("Fuzzy C-Means Experiment GUI")

        self.frame = ttk.LabelFrame(master, text="FCM Configuration", padding=10)
        self.frame.pack(padx=10, pady=10, fill="x")

        # Cluster count
        ttk.Label(self.frame, text="Number of Clusters (K):").grid(row=0, column=0, sticky="w")
        self.k_entry = ttk.Entry(self.frame)
        self.k_entry.insert(0, "4")
        self.k_entry.grid(row=0, column=1)

        # Fuzziness
        ttk.Label(self.frame, text="Fuzziness (m):").grid(row=1, column=0, sticky="w")
        self.m_entry = ttk.Entry(self.frame)
        self.m_entry.insert(0, "2.0")
        self.m_entry.grid(row=1, column=1)

        # Number of runs
        ttk.Label(self.frame, text="Number of Runs:").grid(row=2, column=0, sticky="w")
        self.runs_entry = ttk.Entry(self.frame)
        self.runs_entry.insert(0, "10")
        self.runs_entry.grid(row=2, column=1)

        # Initialization method
        ttk.Label(self.frame, text="Initialization Method:").grid(row=3, column=0, sticky="w")
        self.init_method = ttk.Combobox(self.frame, values=["dirichlet", "uniform"], state="readonly")
        self.init_method.current(0)
        self.init_method.grid(row=3, column=1)

        # Diversity strategy
        ttk.Label(self.frame, text="Diversity Strategy:").grid(row=4, column=0, sticky="w")
        self.div_strategy = ttk.Combobox(self.frame, values=["gaussian", "reinitialize"], state="readonly")
        self.div_strategy.current(0)
        self.div_strategy.grid(row=4, column=1)

        # Distance metric
        ttk.Label(self.frame, text="Distance Metric:").grid(row=5, column=0, sticky="w")
        self.dist_metric = ttk.Combobox(self.frame, values=["euclidean", "mahalanobis", "cosine"], state="readonly")
        self.dist_metric.current(0)
        self.dist_metric.grid(row=5, column=1)

        # Run button
        self.run_button = ttk.Button(master, text="Run FCM", command=self.run_fcm_single)
        self.run_button.pack(pady=10)

        # Status label
        self.result_label = ttk.Label(master, text="", justify="center")
        self.result_label.pack(pady=5)

    def run_fcm_single(self):
        try:
            K = int(self.k_entry.get())
            m = float(self.m_entry.get())
            runs = int(self.runs_entry.get())
            init = self.init_method.get()
            diversity = self.div_strategy.get()
            dist = self.dist_metric.get()
        except ValueError:
            messagebox.showerror("Input Error", "Please provide valid numeric inputs.")
            return

        self.result_label.config(text="Running FCM...")
        self.master.update()

        best_obj = float('inf')
        best_seed = None
        best_centroids = None
        best_U = None

        for _ in range(runs):
            import numpy as np
            seed = np.random.randint(0, 1e6)
            obj, centroids, U = run_fcm(
                X, K, m,
                random_state=seed,
                init_method=init,
                diversity_strategy=diversity,
                distance_metric=dist
            )
            if obj < best_obj:
                best_obj = obj
                best_seed = seed
                best_centroids = centroids
                best_U = U

        self.result_label.config(
            text=f"Best Run:\nObj: {best_obj:.4f} | Seed: {best_seed}"
        )

        plot_best_clusters(X, best_U, best_centroids, best_obj, best_seed, (init, diversity, dist))


# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = FCMApp(root)
    root.mainloop()
