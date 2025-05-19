# 🧠 Customer Segmentation using FCM, GA, and K-Means

This repository presents a clustering-based customer segmentation system that applies both traditional and evolutionary computational intelligence methods, namely:

- **K-Means Clustering**
- **Fuzzy C-Means (FCM)**
- **Genetic Algorithm (GA)-based Clustering**

---

## 🎯 Project Objective

To enhance the performance of customer segmentation through:
- Traditional and fuzzy clustering
- Evolutionary computing using Genetic Algorithms
- Visual comparison of clustering performance across different configurations and metrics

---

## 🗃️ Dataset

- **Name:** Mall Customers Dataset
- **Source:** [Kaggle - Mall Customers Dataset](https://www.kaggle.com/datasets/shwetabh123/mall-customers)
- **Records:** 200 customer entries
- **Attributes:**
  - CustomerID
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1–100)
- **Preprocessing:** Normalization and encoding

---

## 📊 Algorithms Implemented

### 🔹 K-Means Clustering
- Centroid-based partitioning algorithm
- Fast and simple but sensitive to initialization
- Used as a baseline clustering method

### 🔸 Fuzzy C-Means (FCM)
- Allows soft clustering — data points can belong to multiple clusters with varying membership values
- Controlled by fuzzifier constant \( m \)
- Iteratively updates cluster centers and membership matrix

### 🧬 Genetic Algorithm (GA)-Based Clustering
- Evolves cluster centroids using Genetic Algorithms
- Multiple operator variations implemented:
  - **Parent Selection:** Tournament, Roulette Wheel, Random
  - **Crossover Operators:** Arithmetic, Uniform
  - **Mutation Strategies:** Gaussian, Swap, Reset
- Diversity preservation via immigrant strategy
- Parameters tuned:
  - Population size
  - Mutation rate
  - Number of generations

---

## 🧪 Evaluation Metrics

- **Silhouette Score**
- **Fitness Values (GA)**
- **Cluster Visualization (2D/3D)**

Each GA configuration was tested across **30 random seeds**, and results were:
- Logged in CSV files
- Visualized for comparative analysis

---

## 📈 Results Overview

- **GA variants outperformed K-Means** in avoiding local minima and improving cluster compactness
- **FCM** provided more flexible assignments and was helpful when data boundaries were ambiguous
- Cluster plots revealed clear segment separations and customer behavior groupings

---

## 🧰 Tools & Technologies

- **Language:** Python 3.10
- **IDE:** Jupyter Notebook
- **Libraries:**
  - `numpy`, `pandas`
  - `scikit-learn`, `matplotlib`, `seaborn`
  - `scipy` (for FCM)
  - `random`, `copy` (for GA logic)

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-segmentation-ci.git
cd customer-segmentation-ci
