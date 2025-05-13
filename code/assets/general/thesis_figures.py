import numpy as np
import matplotlib.pyplot as plt
import cv2

# function to plot noise schedules
def noise_schedules():
    t = np.linspace(0, 1, 1000)
    beta_min, beta_max = 0.1, 20.0
    beta_t = beta_min + t * (beta_max - beta_min)
    integral = np.cumsum(beta_t) * (t[1] - t[0]) 
    alpha_t = np.exp(-integral)
    snr_vp_linear = alpha_t / (1 - alpha_t)

    s = 0.008  # Small offset to prevent division by zero
    alpha_t_cosine = np.cos((t + s) / (1 + s) * (np.pi / 2)) ** 2 / np.cos(s / (1 + s) * (np.pi / 2)) ** 2
    snr_vp_cosine = alpha_t_cosine / (1 - alpha_t_cosine)

    sigma_min, sigma_max = 0.1, 10.0
    sigma_t = sigma_min * (sigma_max / sigma_min) ** t
    snr_ve = 1 / (sigma_t ** 2)

    plt.figure(figsize=(6, 4))
    plt.plot(t, np.log(snr_vp_linear), label="VP-Linear", linestyle="-")
    plt.plot(t, np.log(snr_vp_cosine), label="VP-Cosine", linestyle="-")
    plt.plot(t, np.log(snr_ve), label="Variance Exploding", linestyle="-.")
    plt.xlabel("Time (t)")
    plt.ylabel("log(SNR)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-10, 10)
    plt.show()


def mix_with_gaussian(image, step, total_steps):
    alpha = 1 - (step / (total_steps - 1)) 
    noise = np.random.normal(127.5, 50, image.shape) 
    blended_image = alpha * image + (1 - alpha) * noise 
    return np.clip(blended_image, 0, 255).astype(np.uint8)


def diffusion_forward_process(image_path, steps=7):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image at '{image_path}'. Please check the file path.")
    image = cv2.resize(image, (256, 256)) 
    image = image.astype(np.float32)
    fig, axes = plt.subplots(1, steps, figsize=(15, 5))
    for i in range(steps):
        noisy_image = mix_with_gaussian(image, i, steps)
        axes[i].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
    plt.show()

def score_fields():
    x, y, z = np.meshgrid(np.linspace(-5, 5, 6), 
                        np.linspace(-5, 5, 6),
                        np.linspace(-5, 5, 6))
    mode1 = np.array([-3, -3, -3])
    mode2 = np.array([3, 3, 3])
    u1 = np.zeros_like(x)
    v1 = np.zeros_like(y)
    w1 = np.zeros_like(z)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                pos = np.array([x[i,j,k], y[i,j,k], z[i,j,k]])
                vec1 = mode1 - pos
                vec2 = mode2 - pos
                u1[i,j,k], v1[i,j,k], w1[i,j,k] = vec1/np.linalg.norm(vec1)**2 + vec2/np.linalg.norm(vec2)**2

    u2 = np.random.rand(*x.shape) * 2 - 1 
    v2 = np.random.rand(*y.shape) * 2 - 1
    w2 = np.random.rand(*z.shape) * 2 - 1
    magnitudes = np.sqrt(u2**2 + v2**2 + w2**2)
    u2 = u2 / magnitudes
    v2 = v2 / magnitudes
    w2 = w2 / magnitudes
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.quiver(x, y, z, u1, v1, w1, length=1,arrow_length_ratio=0.5, normalize=True, color='blue')
    ax1.set_title(r'$\nabla_{\mathbf{x}}\log p(\mathbf{x})$', fontsize=25)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(x, y, z, u2, v2, w2, length=1, arrow_length_ratio=0.5,color='orange')
    ax2.set_title(r'$\hat{\mathbf{s}}_{\boldsymbol{\theta}}(\mathbf{x})$', fontsize=25)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    plt.tight_layout()
    plt.show()   
