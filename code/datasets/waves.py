import numpy as np
import matplotlib.pyplot as plt

# Generate Waves
def generate_parallel_waves(num_samples=100000, noise=0.03):
    """ Generate two parallel waves """
    num_samples_per_wave = num_samples // 2
    x = np.linspace(0, 1, num_samples_per_wave)
    y1 = 0.6 + 0.3 * np.sin(2 * np.pi * 3 * x)
    y2 = 0.3 + 0.3 * np.sin(2 * np.pi * 3 * x)
    y1 += np.random.normal(0, noise, num_samples_per_wave)
    y2 += np.random.normal(0, noise, num_samples_per_wave)
    wave1 = np.vstack((x, y1)).T
    wave2 = np.vstack((x, y2)).T
    x = np.vstack((wave1, wave2))
    np.random.shuffle(x)

    # Scaling
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    x = (x - mean_x)/std_x
    return x

# Visualize the dataset
def plot_waves(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.7, color='dodgerblue')
    plt.title('Parallel Sinusoidal Waves')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

