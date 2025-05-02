import numpy as np
import matplotlib.pyplot as plt

def generate_intersecting_rings(center1=(0, 0), center2=(5, 0), num_samples_per_ring=50000, radius=4, width=0.4, noise=0):
    """ Generate a dataset consisting of two intersecting rings. """
    # Generate the first ring
    angles1 = np.random.uniform(0, 2 * np.pi, num_samples_per_ring)
    radii1 = np.random.normal(radius, width / 2, num_samples_per_ring)
    x1 = radii1 * np.cos(angles1) + center1[0]
    y1 = radii1 * np.sin(angles1) + center1[1]
    x1 += np.random.normal(0, noise, num_samples_per_ring)
    y1 += np.random.normal(0, noise, num_samples_per_ring)

    # Generate the second ring
    angles2 = np.random.uniform(0, 2 * np.pi, num_samples_per_ring)
    radii2 = np.random.normal(radius, width / 2, num_samples_per_ring)
    x2 = radii2 * np.cos(angles2) + center2[0]
    y2 = radii2 * np.sin(angles2) + center2[1]
    x2 += np.random.normal(0, noise, num_samples_per_ring)
    y2 += np.random.normal(0, noise, num_samples_per_ring)

    # Combine the two rings
    x = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))
    np.random.shuffle(x)

    # Scaling
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    x = (x - mean_x)/std_x
    return x


# Visualization
def plot_rings(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, color='tomato', alpha=0.6)  # Small markers for clarity
    plt.title("Intersecting Rings")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

