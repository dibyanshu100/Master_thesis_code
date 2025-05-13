import numpy as np
import matplotlib.pyplot as plt

def generate_swiss_roll(n_samples=100000, noise=0.5, hole=False):
    """Generate a swiss roll dataset"""

    generator = np.random.mtrand._rand

    if not hole:
        t = 1.5 * np.pi * (1 + 2 * generator.uniform(size=n_samples))
        y = 21 * generator.uniform(size=n_samples)
    else:
        corners = np.array(
            [[np.pi * (1.5 + i), j * 7] for i in range(3) for j in range(3)]
        )
        corners = np.delete(corners, 4, axis=0)
        corner_index = generator.choice(8, n_samples)
        parameters = generator.uniform(size=(2, n_samples)) * np.array([[np.pi], [7]])
        t, y = corners[corner_index].T + parameters

    x = t * np.cos(t)
    z = t * np.sin(t)

    x = np.vstack((x, y, z))
    x += noise * generator.standard_normal(size=(3, n_samples))
    x = x.T
    x = np.array(x[:, [0,2]])
    t = np.squeeze(t)

    # Scaling
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    x = (x - mean_x)/std_x
    return x


def plot_swiss(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.7, color='forestgreen')
    plt.title('Swiss Roll')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

