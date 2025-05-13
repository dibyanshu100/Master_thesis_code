import os
import json
import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import wasserstein_distance
from ot.sliced import sliced_wasserstein_distance
from sklearn.mixture import GaussianMixture
from os.path import dirname, exists, join
#plt.rcParams.update({'font.size': 13})

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def standard_scaling(data):
    mean = np.mean(data)
    std = np.std(data)
    scaled_data = (data-mean)/std
    return scaled_data

def make_batches(data, device, batch_size, shuffle=False):
    tensor_data = torch.tensor(data).float()
    tensor_data = tensor_data.to(device)
    data_loader = DataLoader(TensorDataset(tensor_data,tensor_data), batch_size = batch_size, shuffle=shuffle)
    return data_loader

def split_train_test(data, test_ratio=0.2): 
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    test_size = int(len(data) * test_ratio)
    test_set = shuffled_data[:test_size]
    train_set = shuffled_data[test_size:]
    return train_set, test_set

def save_and_showfig(fname, show_figure = False):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()

def plot_loss_vs_time(loss_dict, epochs, filename):
    sub_folder, file = filename.split("_", 1)[0], "_".join(filename.split("_", 1)[1:])
    plt.figure(figsize=(6, 4))
    x_axis = [x / 100 for x in loss_dict.keys()]
    plt.plot(x_axis, list(loss_dict.values()))
    plt.xlabel('Time(t)')
    plt.ylabel('Average Loss')
    plt.title(f'Average Loss vs Time Step over {epochs} epochs')
    save_and_showfig(f"results/{sub_folder}/Loss_vs_time_{file}.png", show_figure = True)
    
def plot_mean_var(time_steps, means, variances, filename):
    sub_folder, file = filename.split("_", 1)[0], "_".join(filename.split("_", 1)[1:])
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    # Plot Mean
    ax[0].plot(time_steps, means[:, 0], label='Dim 1')
    ax[0].plot(time_steps, means[:, 1], label='Dim 2')
    ax[0].set_title('Mean of predictions at Each Time Step')
    ax[0].set_xlabel('Time Step (t)')
    ax[0].set_ylabel('Mean')
    ax[0].legend()
    # Plot Variance
    ax[1].plot(time_steps, variances[:, 0], label='Dim 1')
    ax[1].plot(time_steps, variances[:, 1], label='Dim 2')
    ax[1].set_title('Variance of predictions at Each Time Step')
    ax[1].set_xlabel('Time Step (t)')
    ax[1].set_ylabel('Variance')
    ax[1].legend()
    save_and_showfig(f"results/{sub_folder}/Mean_Var_Predictions_{file}.png", show_figure = True)

def plot_generated_samples(data, step_counts, filename):
    sub_folder, file = filename.split("_", 1)[0], "_".join(filename.split("_", 1)[1:])
    if "Waves" in filename:
        color = "dodgerblue"
    elif "Ring" in filename:
        color = "tomato"
    elif "Swiss" in filename:
        color = "forestgreen"
    elif "Cluster" in filename:
        color = "goldenrod"
    else:
        color = "black"  

    fig, axs = plt.subplots(1, 9, figsize=(45, 5)) 
    for k in range(9):  
        axs[k].scatter(data[k, :, 0], data[k, :, 1], color=color)
        axs[k].set_title(f"Steps={step_counts[k]}", fontsize=40)
        axs[k].tick_params(labelsize=30)
    plt.tight_layout()
    save_and_showfig(f"results/{sub_folder}/{file}.png", show_figure = True)

def plot_all_loss_by_t(loss_by_t_list, epochs, filename, scale_type):
    plt.figure(figsize=(7, 5))
    colors = ['gold', 'tomato', 'forestgreen', 'dodgerblue']
    x_axis = [x / 100 for x in loss_by_t_list[0].keys()]
    plt.plot(x_axis, list(loss_by_t_list[0].values()), color = colors[0], label = 'cluster_data')
    plt.plot(x_axis, list(loss_by_t_list[1].values()), color = colors[1], label = 'ring_data')
    plt.plot(x_axis, list(loss_by_t_list[2].values()), color = colors[2], label = 'swiss_roll_data')
    plt.plot(x_axis, list(loss_by_t_list[3].values()), color = colors[3], label = 'waves_data')
    plt.xlabel('Time(t)')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.title(f'Average Loss vs Time Step over {epochs} epochs')
    save_and_showfig(f"results/Loss_vs_Time/{scale_type}/{filename}.png", show_figure = True)

def plot_all_loss_by_epochs(loss_by_epochs_list, epochs, filename, scale_type):
    plt.figure(figsize=(7, 5))
    colors = ['gold', 'tomato', 'forestgreen', 'dodgerblue']
    num_epochs = len(loss_by_epochs_list[0])
    x_axis = np.linspace(0, num_epochs, len(loss_by_epochs_list[0]))
    plt.plot(x_axis, loss_by_epochs_list[0], color=colors[0], label='cluster_data')
    plt.plot(x_axis, loss_by_epochs_list[1], color=colors[1], label='ring_data')
    plt.plot(x_axis, loss_by_epochs_list[2], color=colors[2], label='swiss_roll_data')
    plt.plot(x_axis, loss_by_epochs_list[3], color=colors[3], label='waves_data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss vs Epochs after {epochs} epochs')
    save_and_showfig(f"results/Loss_vs_Epochs/{scale_type}/{filename}.png", show_figure = True)


def plot_all_datasets(data_dict, fname, save= False):
    titles = list(data_dict.keys())
    datasets = list(data_dict.values())
    colors = ['gold', 'tomato', 'forestgreen', 'dodgerblue']
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  
    axs = axs.ravel() 
    for i, data in enumerate(datasets):
        axs[i].scatter(data[:, 0], data[:, 1], s=10, alpha=0.7, color=colors[i])
        axs[i].set_title(titles[i], fontsize=23)
        axs[i].grid(True)
        axs[i].axis('equal')
        axs[i].tick_params(axis='both', labelsize=18)

    #fig.suptitle("2D Datasets", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:  
        plt.savefig(fname)
    plt.show()

def save_model(directory, file_name , model):
    file_path = os.path.join(directory, file_name)
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), file_path)

def calculate_entropy(data, n_components=50, grid_size=1000):
    # Step 1: Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(data)
    
    # Step 2: Create a grid of points to evaluate the PDF
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    grid_x, grid_y = np.meshgrid(
        np.linspace(mins[0], maxs[0], grid_size),
        np.linspace(mins[1], maxs[1], grid_size)
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Step 3: Evaluate the PDF on the grid
    log_pdf = gmm.score_samples(grid_points)  # Log probability density
    pdf = np.exp(log_pdf)  # Convert to probability density

    # Step 4: Approximate the integral (sum over grid points)
    dx = (maxs[0] - mins[0]) / grid_size
    dy = (maxs[1] - mins[1]) / grid_size
    entropy = -np.sum(pdf * log_pdf) * dx * dy

    return entropy

def compare_datasets_entropy(datasets, component_values=[5], grid_size=1000):
    dataset_names = list(datasets.keys())
    dataset_values = list(datasets.values())
    x_positions = np.arange(len(datasets))  # X-axis positions

    # Assign colors based on dataset names
    colors = [
        "goldenrod" if "cluster" in name else
        "tomato" if "ring" in name else
        "forestgreen" if "swiss" in name else
        "dodgerblue" if "waves" in name else "black"
        for name in dataset_names
    ]

    plt.figure(figsize=(7, 4))

    for n_components in component_values:
        entropy_values = [calculate_entropy(data, n_components, grid_size) for data in dataset_values]
        plt.scatter(x_positions, entropy_values, label=f'n_components={n_components}', color=colors, s=20)
        plt.plot(x_positions, entropy_values, linestyle='-', color='gray', alpha=0.5)

    # Customize x-axis
    plt.xticks(x_positions, dataset_names)  # Use dataset names for x-axis labels
    plt.title("Entropy Comparison Across Datasets")
    plt.xlabel("Datasets")
    plt.ylabel("Entropy")
    plt.legend(title="n_components", fontsize=8)
    plt.grid(True)
    filepath = f"results/Entropy_Comparison.png"
    plt.savefig(filepath)
    plt.show()

def plot_noise_at_different_levels(data_dict, filename_prefix="output"):
    
    time_steps = np.linspace(0, 1, 10)  
    colors = {
        "waves_data": "dodgerblue",
        "ring_data": "tomato",
        "swiss_roll_data": "forestgreen",
        "cluster_data": "goldenrod"
    }
    data_dict = {key: data[:10000] for key, data in data_dict.items()}
    
    fig, axs = plt.subplots(len(data_dict), 10, figsize=(40, len(data_dict) * 5)) 

    for row_idx, (key, data) in enumerate(data_dict.items()):
        color = colors.get(key, "black") 
        for col_idx, t in enumerate(time_steps):
            data = torch.tensor(data)
            ts = torch.full((data.size(0), 1), t)
            alpha_t = torch.cos((torch.pi / 2) * ts)
            sigma_t = torch.sin((torch.pi / 2) * ts)
            noise = torch.randn_like(data)
            data_t = alpha_t * data + sigma_t * noise
            
            # Compute Wasserstein distance
            data_np = data_t.cpu().numpy().flatten()  # Flatten data for comparison
            gaussian_samples = np.random.normal(0, 1, len(data_np))  # Reference Gaussian
            wasserstein_dist = wasserstein_distance(data_np, gaussian_samples)

            # Update title with Wasserstein distance
            axs[row_idx, col_idx].scatter(data_t[:, 0].cpu(), data_t[:, 1].cpu(), color=color, s=10)
            axs[row_idx, col_idx].set_title(
                f"{key} | t={t:.2f}\nWDist={wasserstein_dist:.2f}"
            )
            axs[row_idx, col_idx].set_xlim([-3, 3]) 
            axs[row_idx, col_idx].set_ylim([-3, 3])
            axs[row_idx, col_idx].axis("off")

    plt.tight_layout()
    filepath = f"results/{filename_prefix}_grid.png"
    plt.savefig(filepath)
    plt.show()


def plot_scaled_vs_unscaled(unscaled, scaled):
    labels = ['x', 'epsilon', 'v', 'score']
    keys = ['diffusion_loss_x', 'diffusion_loss_epsilon', 'diffusion_loss_v', 'diffusion_loss_score']
    _, axes = plt.subplots(2, 2, figsize=(8, 6))
    for i, ax in enumerate(axes.flat):
        key = keys[i]
        ax.plot(unscaled[key], label='Unscaled', marker='o', color='blue')
        ax.plot(scaled[key], label='Scaled', marker='x', color='red')
        ax.set_title(labels[i])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Value')
        ax.legend()

    plt.tight_layout()
    filepath = f"results/Scaled_vs_Unscaled_loss.png"
    plt.savefig(filepath)
    plt.show()


def hellinger_distance(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

def find_corruption_rate_hellinger(data_dict, num_time_steps=100):
    time_steps = np.linspace(0, 1, num_time_steps)
    distances = {key: [] for key in data_dict.keys()}
    for key, data in data_dict.items():
        data = torch.tensor(data[:10000])
        for t in time_steps:
            ts = torch.full((data.size(0), 1), t)
            alpha_t = torch.cos((torch.pi / 2) * ts)
            sigma_t = torch.sin((torch.pi / 2) * ts)
            noise = torch.randn_like(data)
            data_t = alpha_t * data + sigma_t * noise
            data_np = data_t.cpu().numpy().flatten()
            gaussian_samples = np.random.normal(0, 1, len(data_np))
            data_hist, bins = np.histogram(data_np, bins=100, density=True)
            gaussian_hist, _ = np.histogram(gaussian_samples, bins=bins, density=True)
            data_hist /= np.sum(data_hist)
            gaussian_hist /= np.sum(gaussian_hist)
            hellinger_dist = hellinger_distance(data_hist, gaussian_hist)
            distances[key].append(hellinger_dist)

    # Analyze corruption rate
    corruption_rates = {}
    range_size = 0.1
    num_ranges = int(1 / range_size)
    for key, dists in distances.items():
        corruption_rates[key] = {}
        for i in range(num_ranges):
            start_idx = int(i * range_size * num_time_steps)
            end_idx = int((i + 1) * range_size * num_time_steps)
            dist_range = dists[start_idx:end_idx]
            range_diff = np.diff(dist_range)
            corruption_rate = np.abs(np.sum(range_diff))/(range_size * num_time_steps)
            corruption_rates[key][f"{i * range_size:.1f}-{(i + 1) * range_size:.1f}"] = corruption_rate
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    index = np.arange(len(list(corruption_rates["waves_data"].keys())))
    colors = ['blue', 'green', 'orange', 'red']
    for i, (key, rate_dict) in enumerate(corruption_rates.items()):
        ranges = list(rate_dict.keys())
        corruption_vals = list(rate_dict.values())
        plt.bar(index + i * bar_width, corruption_vals, bar_width, label=key, color=colors[i])
    plt.xlabel('Time Range')
    plt.ylabel('Average Change in hellinger_distance')
    plt.title('Rate of data corruption over time')
    plt.xticks(index + bar_width * 1.5, list(corruption_rates["waves_data"].keys()))  # Set x-ticks to time ranges
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_corruption_rate_wsd(data_dict, num_time_steps=100):
    time_steps = np.linspace(0, 1, num_time_steps)
    distances = {key: [] for key in data_dict.keys()}
    for key, data in data_dict.items():
        data = torch.tensor(data[:10000])
        for t in time_steps:
            ts = torch.full((data.size(0), 1), t)
            alpha_t = torch.cos((torch.pi / 2) * ts)
            sigma_t = torch.sin((torch.pi / 2) * ts)
            noise = torch.randn_like(data)
            data_t = alpha_t * data + sigma_t * noise
            data_np = data_t.cpu().numpy().flatten()
            gaussian_samples = np.random.normal(0, 1, len(data_np))
            wasserstein_dist = np.round(wasserstein_distance(data_np, gaussian_samples),5)
            distances[key].append(wasserstein_dist)

    # Corruption rate
    corruption_rates = {}
    range_size = 0.1
    num_ranges = int(1 / range_size)
    
    for key, dists in distances.items():
        corruption_rates[key] = {}
        for i in range(num_ranges):
            start_idx = int(i * range_size * num_time_steps)
            end_idx = int((i + 1) * range_size * num_time_steps)
            dist_range = dists[start_idx:end_idx]
            range_diff = np.diff(dist_range)
            corruption_rate = np.abs(np.sum(range_diff))/(range_size * num_time_steps)
            corruption_rates[key][f"{i * range_size:.1f}-{(i + 1) * range_size:.1f}"] = corruption_rate

    plt.figure(figsize=(12, 6))
    bar_width = 0.2  
    index = np.arange(len(list(corruption_rates["waves_data"].keys())))
    colors = ['blue', 'green', 'orange', 'red']
    for i, (key, rate_dict) in enumerate(corruption_rates.items()):
        ranges = list(rate_dict.keys())
        corruption_vals = list(rate_dict.values())
        plt.bar(index + i * bar_width, corruption_vals, bar_width, label=key, color=colors[i])
    plt.xlabel('Time Range')
    plt.ylabel('Average Change in Wasserstein Distance')
    plt.title('Rate of data corruption over time')
    plt.xticks(index + bar_width * 1.5, list(corruption_rates["waves_data"].keys())) 
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_transition(path, data_type, epoch_key='loss_vs_time_per_epoch'):
    ## Loss transition per epoch
    data = load_data(path)
    loss_vs_time_per_epoch = data[data_type][epoch_key]
    
    # Generate a colormap with transitions from light to dark
    cmap = get_cmap('plasma')
    num_epochs = len(loss_vs_time_per_epoch)
    colors = [cmap(i / num_epochs) for i in range(num_epochs)]
    
    plt.figure(figsize=(10, 6))
    
    # Iterate through each epoch and plot the loss over time
    for i, epoch_data in enumerate(loss_vs_time_per_epoch):
        time_steps = list(map(int, epoch_data.keys()))
        losses = list(epoch_data.values())  
        
        #plt.plot(time_steps, losses, label=f'Epoch {i}', color=colors[i])
        plt.plot(time_steps, losses, color=colors[i])
    # Add a color bar to represent the transition of shades across epochs
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_epochs - 1))
    cbar = plt.colorbar(sm, orientation='vertical')
    cbar.set_label('Epochs', rotation=270, labelpad=15)
    
    
    # Plot configuration
    plt.title("Loss vs Time (Transition of Shades Across Epochs)")
    plt.xlabel("Time Steps")
    plt.ylabel("Train Loss")
    plt.legend(title="Epochs")
    plt.grid(alpha=0.3)
    plt.show()


def compute_sliced_wasserstein(data_dict, n_projections=50, plot=False):
    time_steps = np.arange(0.01, 1, 0.1)
    swd_results = {key: [] for key in data_dict.keys()}  
    unit_gaussian_samples = np.random.normal(0, 1, (10000, 2))
    # Define colors for each dataset
    colors = {
        "waves_data": "dodgerblue",
        "ring_data": "tomato",
        "swiss_roll_data": "forestgreen",
        "cluster_data": "goldenrod"
    }

    if plot:
        fig, axes = plt.subplots(len(data_dict), len(time_steps), figsize=(30, 12))

    for row_idx, (dataset_name, data) in enumerate(data_dict.items()):
        for col_idx, t in enumerate(time_steps):
            data = data[:10000]
            ts = torch.full((data.shape[0], 1), t)
            alpha_t = torch.cos((torch.pi / 2) * ts)
            sigma_t = torch.sin((torch.pi / 2) * ts)
            noise = torch.randn_like(torch.tensor(data))
            data_t = alpha_t * torch.tensor(data) + sigma_t * noise
            data_np = data_t.cpu().numpy()

            # Sliced Wasserstein Distance (SWD)
            swd = sliced_wasserstein_distance(data_np, unit_gaussian_samples, n_projections=n_projections)
            swd_results[dataset_name].append(swd)

            if plot:
                # Plot histogram of noisy data
                hist_data, xedges, yedges = np.histogram2d(data_np[:, 0], data_np[:, 1], bins=50, density=True)
                ax = axes[row_idx, col_idx]
                ax.imshow(hist_data.T, origin='lower', aspect='auto', interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(f"{dataset_name}", fontsize=20)
                if row_idx == 0:
                    ax.set_title(f"t={t:.2f}", fontsize=25)
                #ax.set_xlabel("X")
                #ax.set_ylabel("Y")

    # if plot:
    #     # Plot the unit Gaussian distribution in the last row
    #     for col_idx, t in enumerate(time_steps):
    #         hist_gaussian, xedges, yedges = np.histogram2d(unit_gaussian_samples[:, 0], unit_gaussian_samples[:, 1], bins=50, density=True)
    #         ax = axes[len(data_dict), col_idx]
    #         ax.imshow(hist_gaussian.T, origin='lower', aspect='auto', interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    #         if col_idx == 0:
    #             ax.set_ylabel("Unit Gaussian", fontsize=12)
    #         ax.set_xlabel("X")
    #         ax.set_ylabel("Y")

    plt.figure(figsize=(10, 5))
    
    bar_width = 0.2  # Width of each bar
    index = np.arange(len(time_steps) - 1)  # Positions for x-ticks
    colors = ['goldenrod', 'tomato', 'forestgreen', 'dodgerblue']  # Colors for each dataset

    for i, (dataset_name, distances) in enumerate(swd_results.items()):
        # Calculate absolute changes in distances between consecutive time steps
        distance_changes = np.abs(np.diff(distances))

        # Plot bars for this dataset
        plt.bar(index + i * bar_width, distance_changes, bar_width, 
                label=dataset_name, color=colors[i])

    # Customize the plot
    plt.xlabel('Time Range')
    plt.ylabel('Change in Sliced Wasserstein Distance')
    plt.title('Change in Sliced Wasserstein Distance Over Time')
    plt.xticks(index + bar_width * (len(swd_results) - 1) / 2, 
               [f"{t1:.2f}→{t2:.2f}" for t1, t2 in zip(time_steps[:-1], time_steps[1:])], rotation=45)
    plt.legend(title="Datasets")
    plt.tight_layout()
    plt.show()
    return swd_results




#######################
###### ImageData ######
#######################

def show_image_samples(samples, fname = None, nrow = 10, title = 'Generated Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title, fontsize = 20)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")

    if fname is not None:
        save_and_showfig(fname)
    else:
        plt.show()

def load_cifar_data():
    train_data = torchvision.datasets.CIFAR10("./datasets/Image_data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    test_data = torchvision.datasets.CIFAR10("./datasets/Image_data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=False)
    return train_data, test_data

def extract_cifar10_images(save_path):
    os.makedirs(save_path, exist_ok=True)
    train_data = torchvision.datasets.CIFAR10("./datasets/Image_data", transform=torchvision.transforms.ToTensor(), download=True, train=True)
    
    for idx, (img, _) in enumerate(train_data):
        img = torchvision.transforms.ToPILImage()(img)
        img.save(os.path.join(save_path, f"img_{idx}.png"))

def visualize_cifar_data():
    train_data, _ = load_cifar_data()
    imgs = train_data.data[:100]
    show_image_samples(imgs, title=f'CIFAR-10 Samples')

def save_training_plot(train_losses, test_losses, title, fname) -> None:
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label="train loss")
    plt.plot(x_test, test_losses, label="test loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    save_and_showfig(fname)

def save_data(data, path):
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj 

    serializable_data = make_serializable(data)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        json.dump(serializable_data, f)

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data
  
def DiffusionImages(
        function, 
        scaling_type, 
        loss_formulation, 
        sampling,
        plot_loss_by_epochs = False, 
        plot_loss_by_time = False, 
        generate_samples = False, 
        save_params = False,
        save_losses = False
        ):
    
    train_data, test_data = load_cifar_data()
    train_data = train_data.data / 255.0
    test_data = test_data.data / 255.0
    train_losses, test_losses, samples, average_loss_by_t, loss_by_t_per_epoch = function(train_data, test_data, scaling_type, loss_formulation, sampling, plot_loss_by_time, save_params)
    
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    fname = f'results/Images/{scaling_type}/' + f'{loss_formulation.__name__}'
    if plot_loss_by_epochs:
        save_training_plot(
            train_losses,
            test_losses,
            "CIFAR Train Plot",
            fname + "_train_plot"
        ) 

    if generate_samples:
        show_image_samples(samples * 255.0, fname+"_samples", title=f"CIFAR-10 generated samples")

    if save_losses:
        loss_dict = {'train_loss':train_losses, 'test_loss':test_losses, 'loss_vs_time':average_loss_by_t, 'loss_vs_time_per_epoch':loss_by_t_per_epoch}
        path = f'saved_data/Images/{scaling_type}/' + f'{loss_formulation.__name__}' + '_results'
        save_data(loss_dict, path)


####################################################

# Results for Paper

## Average of test loss
def calculate_average_final_test_loss(base_path, loss_files, executions):
    avg_final_losses = {
        "Cluster_data": {"epsilon": 0, "x": 0, "v": 0, "score": 0},
        "Ring_data": {"epsilon": 0, "x": 0, "v": 0, "score": 0},
        "Swiss_roll_data": {"epsilon": 0, "x": 0, "v": 0, "score": 0},
        "Waves_data": {"epsilon": 0, "x": 0, "v": 0, "score": 0}
    }

    for exec_num in range(1, executions + 1):
        execution_path = os.path.join(base_path, f"Execution_{exec_num}")
        
        for loss_file in loss_files:
            file_path = os.path.join(execution_path, loss_file)
            data = load_data(file_path)
            for data_type, loss_data in data.items():
                if data_type in avg_final_losses:
                    final_test_loss = loss_data["test_loss"][-1] 
                    avg_final_losses[data_type][loss_file.split('_')[2]] += final_test_loss
    
    for data_type in avg_final_losses:
        for loss_type in ["epsilon", "x", "v", "score"]:
            avg_final_losses[data_type][loss_type] /= executions
     
    return avg_final_losses


## Loss_vs_Epochs 
def plot_loss_across_epochs(base_path, loss_files, executions, loss_type, scale_type):
    loss_data = {
        "Cluster_data": {"x": [], "epsilon": [], "v": [], "score": []},
        "Ring_data": {"x": [], "epsilon": [], "v": [], "score": []},
        "Swiss_roll_data": {"x": [], "epsilon": [], "v": [], "score": []},
        "Waves_data": {"x": [], "epsilon": [], "v": [], "score": []}
    }
    colours = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']

    for exec_num in range(1, executions + 1):
        execution_path = os.path.join(base_path, f"Execution_{exec_num}")
        
        for loss_file in loss_files:
            file_path = os.path.join(execution_path, loss_file)
            data = load_data(file_path)
            
            # For each data type, extract the loss data
            for data_type, loss_data_per_type in data.items():
                if data_type in loss_data:
                    loss = loss_data_per_type[loss_type]
                    loss_key = loss_file.split('_')[2]
                    loss_data[data_type][loss_key].append(loss)
    # Create a 2x2 subplot grid
    _, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    axes = axes.flatten()
    for i, (data_type, loss_functions) in enumerate(loss_data.items()):
        ax = axes[i]
        j=0
        for loss_formulation, loss_values in loss_functions.items():
            loss_values = np.array(loss_values)   # Shape: (executions, epochs)
            avg_loss = np.mean(loss_values, axis=0)
            std_loss = np.std(loss_values, axis=0)
            if loss_formulation == 'x':
                labl = r'$\mathbf{x}$'
            if loss_formulation == 'epsilon':
                labl = r'$\mathbf{\epsilon}$'
            if loss_formulation == 'v':
                labl = r'$\mathbf{v}$'
            if loss_formulation == 'score':
                labl = r'$\mathbf{s}$'
            ax.plot(avg_loss, label=labl, linestyle='-', linewidth=2, color= colours[j])
            ax.fill_between(range(len(avg_loss)), avg_loss - std_loss, avg_loss + std_loss, alpha=0.4)
            j= j+1
        # Set the y-axis to log scale
        ax.set_yscale('log')
        ax.set_title(f"{data_type} - {scale_type}", weight='bold', fontsize=20)
        ax.set_xlabel("Epochs" , fontsize=18)
        ax.set_ylabel(f"{loss_type}", fontsize=18)
        ax.grid(True)
        ax.legend(fontsize=20)
        ax.set_ylim(0, 40)
        ax.tick_params(axis='y', labelleft=True, labelsize=20)
        ax.tick_params(axis='x', labelsize=20)
    plt.tight_layout()
    plt.show()


## Loss vs Time
def plot_loss_by_t_per_epoch(base_path, data_type):
    #epochs_to_plot = [i for i in range(4, 100, 20)] + [99]
    epochs_to_plot = [5, 50, 99]
    loss_files = [
        "diffusion_loss_x_training_results",
        "diffusion_loss_epsilon_training_results",
        "diffusion_loss_v_training_results",
        "diffusion_loss_score_training_results"
    ]
    execution_path = os.path.join(base_path, "Execution_1")
    _, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()
    for i, loss in enumerate(loss_files):
        ax = axes[i]
        data_path = os.path.join(execution_path, loss)
        data = load_data(data_path)
        if data_type in data:
            loss_vs_time_per_epoch = data[data_type]["loss_vs_time_per_epoch"]
            
            for pos in epochs_to_plot:
                if pos < len(loss_vs_time_per_epoch):
                    dictionary = loss_vs_time_per_epoch[pos]
                    timesteps = list(map(int, dictionary.keys()))
                    losses = list(dictionary.values())
                    
                    ax.plot([t / 100 for t in timesteps], losses, label=f"Epoch {pos+1}")
                else:
                    print(f"Position {pos} is out of range for {data_type}.")
        labl = loss.split('_')[2]
        if labl == 'x':
            param = r'$\mathbf{x}$'
            ax.set_title(f"{data_type}-{param}", fontsize=20)
        if labl == 'epsilon':
            param = r'$\mathbf{\epsilon}$'
            ax.set_title(f"{data_type}-{param}", fontsize=20)
        if labl == 'v':
            param = r'$\mathbf{v}$'
            ax.set_title(f"{data_type}-{param}", fontsize=20)
        if labl == 'score':
            param = r'$\mathbf{s}$'
            ax.set_title(f"{data_type}-{param}", fontsize=20)
        ax.set_xlabel("Timestep", fontsize = 20)
        ax.set_ylabel("Loss", fontsize = 20)
        #ax.set_title(f"{data_type}-({loss.split('_')[2]})")
        ax.grid(True)
        ax.legend(fontsize = 16)
        ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.show()


# Equivalent Loss
def plot_equivalent_losses(Equivalent_losses):
    loss_space = ['x', 'epsilon', 'v', 'score'] 
    labels = [r'$\mathbf{x}$',r'$\mathbf{\epsilon}$',r'$\mathbf{v}$',r'$\mathbf{s}$']
    colours = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']  # Corresponding colors for each loss type
    
    # Set up the figure with 4 rows and 1 column
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)  
    axes = axes.flatten()
    data_types = list(Equivalent_losses.keys())  # Extract the keys (data types)
    
    for i, data_type in enumerate(data_types):
        ax = axes[i]  # Get the corresponding axis
        loss_dict = Equivalent_losses[data_type]  # Get the loss dictionary for the current data type
        for j, loss_type in enumerate(loss_space):
            if loss_type in loss_dict:  # Check if the loss type exists in the dictionary
                loss_values = loss_dict[loss_type]
                ax.plot(
                    range(len(loss_values)), 
                    loss_values, 
                    label=labels[j], 
                    color=colours[j]  # Assign color based on the order in `colours`
                )
        
        # Set plot labels and title
        ax.set_yscale('log')
        ax.set_xlabel("Epochs", fontsize=18)
        ax.set_ylabel("test_loss", fontsize=18)
        ax.set_title(f"{data_type} - Rescaled",  weight='bold', fontsize=20)
        ax.grid(True)
        ax.legend(fontsize=20)  # Add a legend to distinguish loss types
        ax.set_ylim(0, 40)
        ax.tick_params(axis='y', labelleft=True, labelsize=20)
        ax.tick_params(axis='x', labelsize=20)
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


# plot ELBO coefficents
def plot_elbo_coefficients():
    t = np.linspace(0, 1, 1000)
    alpha_t = np.cos((np.pi / 2) * t)
    sigma_t = np.sin((np.pi / 2) * t)
    y1 = alpha_t / ((sigma_t**3) + 1e-2)
    y2 = 1 / ((sigma_t * alpha_t) + 1e-2)
    y3 = alpha_t / ((sigma_t * (alpha_t**2 + sigma_t**2)) + 1e-2)
    y4 = sigma_t / ((alpha_t) + 1e-2)
    coefficients = {
        'x-coeff': (y1, 'blue'),
        'ep-coeff': (y2, 'orange'),
        'v-coeff': (y3, 'green'),
        'score-coeff': (y4, 'red')
    }
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, (y, color)) in zip(axs, coefficients.items()):
        ax.plot(t, y, label=title, color=color)
        if title == 'x-coeff':
            ax.set_title(r'$\mathbf{x}-\mathrm{space}$', fontsize=20)
            ax.set_ylabel(r'$\frac{1}{w_{\mathbf{x}}(t)}$', fontsize=25)
        elif title == 'ep-coeff':
            ax.set_title(r'$\mathbf{\epsilon}-\mathrm{space}$', fontsize=20)
            ax.set_ylabel(r'$\frac{1}{w_{\mathbf{\epsilon}}(t)}$', fontsize=25)
        elif title == 'v-coeff':
            ax.set_title(r'$\mathbf{v}-\mathrm{space}$', fontsize=20)
            ax.set_ylabel(r'$\frac{1}{w_{\mathbf{v}}(t)}$', fontsize=25)
        elif title == 'score-coeff':
            ax.set_title(r'$\mathbf{s}-\mathrm{space}$', fontsize=20)
            ax.set_ylabel(r'$\frac{1}{w_{\mathbf{s}}(t)}$', fontsize=25)
        ax.set_xlabel('Time (t)', fontsize=18)
        ax.tick_params(labelsize=16)
        #ax.legend()
        ax.grid(True)
    plt.tight_layout()
    # fig.suptitle('Plot of ELBO Coefficients', fontsize=16, y=1.05)
    plt.show()

# plot samples for all data
def plot_samples(data, datatype):
    losses = [r'$\mathbf{x}$', r'$\mathbf{\epsilon}$', r'$\mathbf{v}$', r'$\mathbf{s}$']
    # Assign colors based on datatype
    if "Waves" in datatype:
        color = "dodgerblue"
    elif "Ring" in datatype:
        color = "tomato"
    elif "Swiss" in datatype:
        color = "forestgreen"
    elif "Cluster" in datatype:
        color = "goldenrod"
    else:
        color = "black"

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    axs = axs.flatten()  
    # Plot data in each subplot
    for k in range(4):
        ax = axs[k]
        ax.scatter(data[k, :, 0], data[k, :, 1], color=color, s=10)  # s=10 sets the marker size
        ax.set_title(f"{losses[k]}-space", fontsize = 20)  # Use losses[k] directly if it's a string or has a __str__ method
        ax.axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def plot_variance_weighted(loss_formulations,data_type):
    base_paths = [
        "saved_data/2Dim/Weighted/Execution_1",
        "saved_data/2Dim/Weighted/Execution_2",
        "saved_data/2Dim/Weighted/Execution_3",
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()

    for idx, loss_formulation in enumerate(loss_formulations):
        combined_variances = {}

        for base_path in base_paths:
            path = f"{base_path}/{loss_formulation.__name__}_training_results"
            results = load_data(path)
            lt_per_epoch = results[data_type]['loss_vs_time_per_epoch']
            
            master_dict = {}
            for di in lt_per_epoch:
                for key, value in di.items():
                    master_dict.setdefault(key, []).append(value)
            
            for key, values in master_dict.items():
                variance = np.var(values)
                combined_variances.setdefault(key, []).append(variance)

        keys = list(combined_variances.keys())
        mean_variances = [np.mean(combined_variances[key]) for key in keys]
        timesteps = np.linspace(0, 1, len(keys))
        ax = axes[idx]
        ax.bar(
            timesteps, 
            mean_variances, 
            color=colors[idx], 
            edgecolor='black', 
            alpha=0.8, 
            width=0.05
        )
        ax.set_xlabel('Timesteps', fontsize=10, labelpad=10)
        ax.set_ylabel('Variance', fontsize=10, labelpad=10)
        ax.set_title(f"{loss_formulation.__name__}", fontsize=10, weight='bold')
        ax.set_xticks(np.linspace(0, 1, 11))  
        ax.set_xticklabels([f"{t:.1f}" for t in np.linspace(0, 1, 11)], rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)  
        ax.spines['top'].set_visible(False)  
        ax.spines['right'].set_visible(False) 

    plt.tight_layout()
    fig.suptitle(f'Variance in objective vs Timesteps - {data_type}', fontsize=12, weight='bold', y=1.02)
    plt.show()
    