import numpy as np
import os
import utils
import loss as l
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from model import Unet
import yaml
from PIL import Image
from torcheval.metrics import FrechetInceptionDistance
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm

with open('config.yaml', "r") as config_file:
  config = yaml.safe_load(config_file)

model_config = config['training_Ndim']['model_params']
train_config = config['training_Ndim']['train_params']
device = model_config['device']

def train_images(train_data, test_data, scaling_type, loss_formulation, sampling, plot_loss_by_time = False, save_params = False):
  
    # Model 
    model = Unet(model_config).to(device)

    # Specify training parameters
    epochs = train_config['num_epochs']
    batch = train_config['batch_size']
    beta1 = train_config['beta1']
    beta2 = train_config['beta2']
    lr = train_config['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr , betas=(beta1, beta2))

    # Scaling
    train_data = torch.from_numpy(train_data) * 2 - 1
    test_data = torch.from_numpy(test_data) * 2 - 1

    # Initialize tracking dictionary for loss vs time_steps
    loss_by_t = {}

    testset = TensorDataset(test_data.float().permute(0, 3, 1, 2).to(device))
    test_loader = DataLoader(testset, batch_size=batch)  
    dataset = TensorDataset(train_data.float().permute(0, 3, 1, 2).to(device))
    train_loader = DataLoader(dataset, batch_size=batch)
    test_loss = []
    train_loss = []
    loss_by_t_per_epoch = []
    for ep in tqdm(range(epochs)):
        epoch_loss_by_t = {k: [] for k in range(100)}
        model.eval()
        with torch.no_grad():
            loss = 0
            for (batch,) in test_loader:
                loss += loss_formulation(batch, model, device, "test", scaling_type, True)
            test_loss.append(loss.item() / len(test_loader))

        model.train()
        for (batch,) in train_loader:
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                loss = loss_formulation(batch, model, device, "train", scaling_type, True, epoch_loss_by_t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss.append(loss.item()/len(train_loader))

        # Anneal Learning Rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (1 - ep / epochs)

        loss_per_epoch = {key: sum(value) / len(value) if value else 0 for key, value in epoch_loss_by_t.items()}
        loss_by_t_per_epoch.append(loss_per_epoch)


    # Aggregate losses for the epoch
    for k, losses in epoch_loss_by_t.items():
        if losses: 
            loss_by_t.setdefault(k, []).append(np.mean(losses))
         
    train_losses = np.array(train_loss)
    test_losses = np.array(test_loss)

    # Calculate the average loss across all epochs per time step
    avg_loss_by_t = {k: np.mean(v) for k, v in loss_by_t.items()}

    # Plot Loss Vs Time
    if plot_loss_by_time:
        utils.plot_loss_vs_time(avg_loss_by_t, epochs, f"Images_{loss_formulation.__name__}_{scaling_type}")

    # Save the model state dictionary
    if save_params:
        utils.save_model(f"saved_models/Images/{scaling_type}/", f"{loss_formulation.__name__}", model) 

    samples = np.zeros((10, 10, 32, 32, 3))
    rand_data = torch.randn((100, 3, 32, 32))
    num_steps =  512
    pred = sampling(model, rand_data, device, num_steps, t_embed=True, clamp_x=True)
    pred = (pred+1)/2.0
    pred = pred.detach().permute(0,2,3,1).cpu().numpy()
    samples = pred.reshape(-1, *samples.shape[2:])
    
    return train_losses, test_losses, samples, avg_loss_by_t, loss_by_t_per_epoch


def run_saved_model_images(state_dict_path, scaling_type, sampling, loss_formulation, num_steps):
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(state_dict_path))
    samples = np.zeros((10, 10, 32, 32, 3))
    rand_data = torch.randn((100, 3, 32, 32))
    print("Number of Steps: ",num_steps)
    pred = sampling(model, rand_data, device, num_steps, t_embed=True, clamp_x=True)
    pred = torch.clamp(pred,-1,1)
    pred = (pred+1)/2.0
    pred = pred.detach().permute(0,2,3,1).cpu().numpy()
    fname = 'results/Images/' + f'{loss_formulation.__name__}' + '_' + scaling_type
    samples = pred.reshape(-1, *samples.shape[2:])
    utils.show_image_samples(samples * 255.0, fname+"_samples", title=f"{scaling_type}")

# samples = np.zeros((10, 10, 32, 32, 3))
#     step_counts =  np.power(2, np.linspace(0, 9, 10)).astype(int)  #[1,2,4,8,16,32,64]
#     print("Step Counts: ", step_counts)
#     for idx, num_steps in enumerate(step_counts):
#        print(num_steps)
#        pred = sampling(model, num_steps)
#        samples[idx] = pred

def run_saved_model_images_different_ts(state_dict_path, scaling_type, sampling, loss_formulation):
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(state_dict_path))
    samples = np.zeros((10, 10, 32, 32, 3))
    rand_data = torch.randn((10, 3, 32, 32))
    step_counts =  np.power(2, np.linspace(0, 9, 10)).astype(int)
    for idx, num_steps in enumerate(step_counts):
        rand_data = torch.randn((10, 3, 32, 32))
        pred = sampling(model, rand_data, device, num_steps, t_embed=True, clamp_x=True)
        pred = torch.clamp(pred,-1,1)
        pred = (pred+1)/2.0
        pred = pred.detach().permute(0,2,3,1).cpu().numpy()
        samples[idx] = pred
    fname = 'results/Images/' + f'{loss_formulation.__name__}' + '_' + scaling_type
    samples = samples.reshape(-1, *samples.shape[2:])
    utils.show_image_samples(samples * 255.0, fname+"_samples", title=f"CIFAR-10 generated samples")



def calculate_fid(samples):
    # Load CIFAR-10 dataset for real images
    # transform = transforms.Compose([
    #     transforms.Resize((299, 299)),  # Inception model expects 299x299 images
    #     transforms.ToTensor(),
    # ])
    samples = samples.to(device)
    real_data, _ = utils.load_cifar_data()
    real_data = torch.from_numpy(real_data.data / 255.0)
    real_data = real_data[0:1000,...]
    dataset = TensorDataset(real_data.float().permute(0, 3, 1, 2).to(device))
    data_loader = DataLoader(dataset, batch_size=124)
    print(len(data_loader))

    fid = FrechetInceptionDistance(device=device)
    fid.real_sum = fid.real_sum.to(torch.float64)
    fid.real_cov_sum = fid.real_cov_sum.to(torch.float64)
    fid.fake_sum = fid.fake_sum.to(torch.float64)
    fid.fake_cov_sum = fid.fake_cov_sum.to(torch.float64)
    
    print("Loading Real data")
    for (real_batch,) in tqdm(data_loader):
        fid.update(real_batch, is_real=True)
    
    sample_dataset = TensorDataset(samples.float().to(device))
    sample_loader = DataLoader(sample_dataset, batch_size=124)
    print("Loading Generated data")
    for (gen_batch,) in tqdm(sample_loader):
        fid.update(gen_batch, is_real=False)
    
    fid_score = fid.compute()
    print(f"FID Score: {fid_score}")
    return fid_score


def generate_images_and_calculate_FID(state_dict_path, sampling, loss_formulation, scaling_type, num_steps, num_images=50000, batch_size=100, save_data = True):
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    
    save_path = f'FID/{scaling_type}/{loss_formulation}_{num_images}'
    num_batches = num_images // batch_size
    image_tensor = torch.empty((num_batches, batch_size, 3, 32, 32), device=device)
    print("Generating images...")
    for batch_idx in tqdm(range(num_batches)):
        rand_data = torch.randn((batch_size, 3, 32, 32)).to(device)
        pred = sampling(model, rand_data, device, num_steps, t_embed=True, clamp_x=True)
        pred = torch.clamp(pred,-1,1)
        pred = (pred + 1) / 2.0
        image_tensor[batch_idx] = pred

    samples = image_tensor.view(num_batches*batch_size, 3, 32, 32)
    #calculate fid    
    fid = calculate_fid(samples).item()
    #samples_list = samples.cpu().numpy().tolist()
    result_dict = {
        "FID": fid,
    }
    
    if save_data:
        utils.save_data(result_dict, save_path)

    return fid, samples

# def calculate_fid(generated_image_dir):
#     print("Calculating FID score...")
#     real_data_path = './datasets/Image_data/real_images'
#     fid = fid_score.calculate_fid_given_paths([real_data_path, generated_image_dir], batch_size=100, device=device, dims=2048)
#     print(f"FID Score: {fid}")
#     return fid
