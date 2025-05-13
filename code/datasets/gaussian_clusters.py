import numpy as np
import matplotlib.pyplot as plt


# Gaussian Clusters
def generate_gaussian_clusters(num_samples_per_cluster=25000, std_dev=1):
    """ Generate a dataset consisting of four Gaussian clusters centered at the specified locations. """
    centers = [(-5, -5), (-5, 5), (5, -5), (5, 5)]
    clusters = [
        np.vstack((
            np.random.normal(center[0], std_dev, num_samples_per_cluster),
            np.random.normal(center[1], std_dev, num_samples_per_cluster)
        )).T
        for center in centers
    ]

    x = np.vstack(clusters)
    np.random.shuffle(x)
    
    # Scaling
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    x = (x - mean_x)/std_x
    return x

# Visualize the dataset
def plot_gaussian_clusters(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, color='gold', alpha=0.6)  # Small markers for clarity
    plt.title("Gaussian Clusters")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()
