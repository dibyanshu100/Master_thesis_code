import os
import numpy as np
import utils
import loss as l
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import Diffusion_Model_2D, Diffusion_Model_2D_Embedded_Time
from utils import make_batches, plot_loss_vs_time, plot_generated_samples, split_train_test, plot_all_loss_by_t, plot_all_loss_by_epochs, plot_mean_var, save_model
import sampling as sample
import yaml
from tqdm import tqdm

with open('config.yaml', "r") as config_file:
  train_config = yaml.safe_load(config_file)

config = train_config["training_2D"]

def train_2D(train_data, test_data, data_type, scale_type, loss_formulation, t_embed= False, sampling = None, plot_loss_by_time = False, plot_samples = False, plot_mean_var_pred = False, save_params = False):
  batch = config['batch_size']
  epochs = config['epochs']
  learning_rate = config['learning_rate']
  hidden_dim = config['hidden_dim']
  x_dim = config['x_dim']
  hidden_layers = config['hidden_layers']
  device = config['device']
  if t_embed:
    temb_dim = config['time_emb_dim']
    model = Diffusion_Model_2D_Embedded_Time(x_dim+temb_dim, hidden_layers, temb_dim, hidden_dim, x_dim)
  else:  
    model = Diffusion_Model_2D(x_dim+1, hidden_layers, hidden_dim, x_dim)
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Batches
  train_loader = make_batches(train_data, device, batch_size = batch, shuffle=True)
  test_loader = make_batches(test_data, device, batch_size = batch, shuffle = False)

  # Initialize tracking dictionary for loss vs time_steps
  loss_by_t = {}

  # Training
  train_epoch_losses = []
  test_epoch_losses = []
  loss_by_t_per_epoch = []
  
  for _ in tqdm(range(epochs)):
    epoch_loss_by_t = {k: [] for k in range(100)}
    train_batch_loss = 0
    test_batch_loss = 0
    model.eval()
    for batch,_ in test_loader:
        loss= loss_formulation(batch, model, device, "test", scale_type, t_embed)
        test_batch_loss += loss.item()

    model.train()
    for batch,_ in train_loader:
        optimizer.zero_grad()
        loss = loss_formulation(batch, model, device, "train", scale_type, t_embed, epoch_loss_by_t)
        loss.backward()
        optimizer.step()
        train_batch_loss += loss.item()
    
    # Aggregate losses for the epoch
    for k, losses in epoch_loss_by_t.items():
        if losses:
            loss_by_t.setdefault(k, []).append(np.mean(losses))

    loss_per_epoch = {key: sum(value) / len(value) if value else 0 for key, value in epoch_loss_by_t.items()}
    loss_by_t_per_epoch.append(loss_per_epoch)
    train_epoch_losses.append(train_batch_loss/len(train_loader))
    test_epoch_losses.append(test_batch_loss/len(test_loader))
  
  # Calculate the average loss across all epochs per time step
  avg_loss_by_t = {k: np.mean(v) for k, v in loss_by_t.items()}

  train_losses = np.array(train_epoch_losses)
  test_losses = np.array(test_epoch_losses)

  # Plot Samples
  if plot_samples:
    all_samples = np.zeros((9, 2000, 2))
    step_counts = np.power(2, np.linspace(0, 9, 9)).astype(int)
    rand_data = torch.randn((2000,2))
    for idx, num_steps in enumerate(step_counts):
        samples = sampling(model, rand_data, device, num_steps, t_embed)
        all_samples[idx] = samples.detach().cpu().numpy()
    plot_generated_samples(all_samples, step_counts, f"{data_type}_{loss_formulation.__name__}_all_samples")

  # Plot Loss Vs Time
  if plot_loss_by_time:
     plot_loss_vs_time(avg_loss_by_t, epochs, f"{data_type}_{loss_formulation.__name__}")

  # Compute/plot mean and Variance of Predictions for trained model
  if plot_mean_var_pred:
     time_steps, means, variances = compute_mean_variance(model, device, batch, t_embed)
     plot_mean_var(time_steps, means, variances, f"{data_type}_{loss_formulation.__name__}")

  # Save the model
  if save_params:
     save_model(f"saved_models/2Dim/{scale_type}/{data_type}", f"{loss_formulation.__name__}_model", model)    
  
  return train_losses, test_losses, avg_loss_by_t, loss_by_t_per_epoch


def save_results(loss_formulation, data_dict, scale_type, run_num = 0, save_data_dict = False, plot_loss = False):
    """ Function to get Loss vs Epochs and loss vs time for each data set"""
    loss_vs_epochs_lists = []
    average_loss_over_time = []
    loss_dict = {}
    for key, val in data_dict.items():
        train_data, test_data = split_train_test(val)
        data_type = key
        train_loss, test_loss, loss_vs_time, loss_vs_time_per_epoch = train_2D(train_data, test_data, data_type, scale_type, loss_formulation)
        loss_dict[data_type] = {'train_loss':train_loss, 'test_loss':test_loss, 'loss_vs_time':loss_vs_time, 'loss_vs_time_per_epoch':loss_vs_time_per_epoch}
        loss_vs_epochs_lists.append(train_loss)
        average_loss_over_time.append(loss_vs_time)

    if plot_loss:
        plot_all_loss_by_epochs(loss_vs_epochs_lists,config['epochs'], f"{loss_formulation.__name__}", scale_type)
        plot_all_loss_by_t(average_loss_over_time,config['epochs'], f"{loss_formulation.__name__}", scale_type)
    # save the dictionary
    if save_data_dict:
        path = f'saved_data/2Dim/{scale_type}/Execution_{run_num}/{loss_formulation.__name__}_training_results'
        utils.save_data(loss_dict, path)

def loss_scaled_unscaled(loss_formulation, data, data_type):
    """ Function to plot ELBO and unscaled loss for different datasets"""
    train_loss_li = []
    test_loss_li = []
    scale_types = ['ELBO', 'no_scaling']
    train_data, test_data = split_train_test(data)
    fname = f'results/{data_type}/{loss_formulation.__name__}_ELBO_vs_Unscaled'
    for scale in scale_types:
       t_embed = True if scale == 'ELBO' else False
       train_loss, test_loss, _ ,_= train_2D(train_data, test_data, data_type, scale, loss_formulation, t_embed)
       train_loss_li.append(train_loss)
       test_loss_li.append(test_loss)
       print(f"Final train loss for {scale} in {data_type} data = {train_loss[-1]}")
       print(f"Final test loss for {scale} in {data_type} = {test_loss[-1]}")
    _, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    # Plot training losses
    axs[0].plot(train_loss_li[0], label='ELBO')
    axs[0].plot(train_loss_li[1], label='Unscaled')
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    # Plot testing losses
    axs[1].plot(test_loss_li[0], label='ELBO')
    axs[1].plot(test_loss_li[1], label='Unscaled')
    axs[1].set_title('Testing Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].legend()
    plt.tight_layout()
    utils.save_and_showfig(fname, show_figure = True)

    

def compute_mean_variance(model, device, data, t_embed, num_time_steps=100):
    """ Function to compute mean and variance of the predictions"""
    model.eval()
    means = []
    variances = []
    time_steps = torch.linspace(0, 1, num_time_steps).to(device)

    with torch.no_grad():
        for t in time_steps:
            t_tensor = torch.full((data.size(0), 1), t).to(device)
            alpha_t = torch.cos((torch.pi/2)*t_tensor).view(-1, *([1] * (data.ndim - 1)))
            sigma_t = torch.sin((torch.pi/2)*t_tensor).view(-1, *([1] * (data.ndim - 1)))
            noise = torch.randn_like(data).to(device)
            x_t = alpha_t * data + sigma_t * noise
            if t_embed:
                predictions = model(x_t,t_tensor.squeeze(-1))
            else:
                input = torch.cat([x_t, t_tensor], dim=1)
                predictions = model(input)
            mean = predictions.mean(dim=0)
            variance = predictions.var(dim=0)
            means.append(mean.detach().cpu().numpy())
            variances.append(variance.detach().cpu().numpy())

    return time_steps.cpu().numpy(), np.array(means), np.array(variances)

def run_saved_model_2D(path, data_type=None, sampling=None, plot_samples= False):
    """Run saved models"""
    config = train_config["training_2D"]
    hidden_dim = config['hidden_dim']
    x_dim = config['x_dim']
    device = config['device'] 
    hidden_layers = config['hidden_layers']
    model = Diffusion_Model_2D(x_dim+1, hidden_layers, hidden_dim, x_dim).to(device)

    # Untrained model
    initial_state = model.state_dict()
    untrained_model = model.__class__(x_dim+1, hidden_layers, hidden_dim, x_dim).to(device)
    untrained_model.load_state_dict(initial_state)

    # Final model
    model.load_state_dict(torch.load(path))
    final_model = model

    if plot_samples:
        all_samples = np.zeros((9, 2000, 2))
        step_counts = np.power(2, np.linspace(0, 9, 9)).astype(int)
        #step_counts = [1,3,5,7,10,40,100,250, 512]
        rand_data = torch.randn((2000,2))
        for idx, num_steps in enumerate(step_counts):
            samples = sampling(final_model, rand_data, device, num_steps)
            all_samples[idx] = samples.detach().cpu().numpy()
        plot_generated_samples(all_samples, step_counts, f"{data_type}_{path.split('/')[-1]}_all_samples")
    
    return untrained_model, final_model

# def evaluate_on_scaled_loss(test_data, model_path):
#     """ Function to compare weighted objectives to equivalent x objectives"""
#     device = config['device']
#     batch = config['batch_size']
#     epochs = config['epochs']
#     untrained_model, final_model= run_saved_model_2D(model_path)
#     test_loader = make_batches(test_data, device, batch_size = batch, shuffle = False)

#     loss_space = ['X', 'EP_X_EQ', 'V_X_EQ', 'SC_X_EQ']
#     test_loss_dict = {}
#     final_model.eval()
#     untrained_model.eval()
#     for scale_type in loss_space:
#         test_epoch_losses = []
#         for epoch in tqdm(range(epochs)):
#             test_batch_loss = 0
#             for batch,_ in test_loader:
#                 if epoch == 0:
#                     loss= l.diffusion_loss_x(batch, untrained_model, device, 'test', scale_type)
#                 else:
#                     loss= l.diffusion_loss_x(batch, final_model, device, 'test', scale_type)
#                 test_batch_loss += loss.item()
#             test_epoch_losses.append(test_batch_loss/len(test_loader))
        
#         test_loss_dict[scale_type] = test_epoch_losses
    
#     return test_loss_dict

def evaluate_on_scaled_loss(test_data, data_name):
    """ Function to compare weighted objectives to equivalent x objectives"""
    base_path = "saved_models/2Dim/Weighted"
    device = config['device']
    batch = config['batch_size']
    epochs = config['epochs']
    test_loader = make_batches(test_data, device, batch_size=batch, shuffle=False)
    test_loss_dict = {}
    
    

    loss_space = ['x', 'epsilon', 'v', 'score']
    models_list = ['diffusion_loss_x_model', 'diffusion_loss_epsilon_model', 'diffusion_loss_v_model', 'diffusion_loss_score_model']
    
    i=0
    for model in models_list:
        model_path = os.path.join(base_path, data_name, f"{model}")
        untrained_model, final_model = run_saved_model_2D(model_path)
        final_model.eval()
        untrained_model.eval()
        test_epoch_losses = []  

        if model == 'diffusion_loss_x_model':
            loss_fn = l.diffusion_loss_x
        if model == 'diffusion_loss_epsilon_model':
            loss_fn = l.diffusion_loss_epsilon
        if model == 'diffusion_loss_v_model':
            loss_fn = l.diffusion_loss_v  
        if model == 'diffusion_loss_score_model':
            loss_fn = l.diffusion_loss_score  
        scale_type = 'Rescaled'

        for epoch in tqdm(range(epochs)):
            test_batch_loss = 0
            for batch, _ in test_loader:
                model = untrained_model if epoch == 0 else final_model
                loss = loss_fn(batch, model, device, 'test', scale_type)
                test_batch_loss += loss.item()
            test_epoch_losses.append(test_batch_loss / len(test_loader))
        
        test_loss_dict[loss_space[i]] = test_epoch_losses
        i=i+1


    # for loss_form in loss_space:
    #     model = models_list[i]
    #     model_path = os.path.join(base_path, data_name, f"{model}")
    #     untrained_model, final_model = run_saved_model_2D(model_path)
    #     test_loader = make_batches(test_data, device, batch_size=batch, shuffle=False)
    #     test_loss_dict = {}
    #     final_model.eval()
    #     untrained_model.eval()
    #     test_epoch_losses = []
        
    #     # Appropriate loss function
    #     loss_fn = getattr(l, f'diffusion_loss_{loss_form}')
    #     print(loss_fn)
    #     scale_type = 'Rescaled'
    #     for epoch in tqdm(range(epochs)):
    #         test_batch_loss = 0
    #         for batch, _ in test_loader:
    #             model = final_model #untrained_model if epoch == 0 else final_model
    #             loss = loss_fn(batch, model, device, 'test', scale_type)
    #             test_batch_loss += loss.item()
    #         test_epoch_losses.append(test_batch_loss / len(test_loader))
        
    #     test_loss_dict[loss_form] = test_epoch_losses
    #     i=i+1
    return test_loss_dict


def Compare_Weighted_losses(data, data_type):
    """ Function to compare weighted objectives to equivalent x objectives"""

    fname = f'results/{data_type}/Comparison_weighted_obj_equivalent_x'
    Weighted_train_loss = {'x':None, 'ep':None, 'v':None, 'sc':None}
    Weighted_test_loss = {'x':None, 'ep':None, 'v':None, 'sc':None}
    Weighted_objectives = [
        ('x', l.diffusion_loss_x), 
        ('ep', l.diffusion_loss_epsilon), 
        ('v', l.diffusion_loss_v), 
        ('sc', l.diffusion_loss_score)
    ]
    train_data, test_data = split_train_test(data)

    ## Train on weighted loss
    for key, loss_formulation in Weighted_objectives:
        scale_type = 'no_scale'
        train_loss, test_loss, _ ,_= train_2D(train_data, test_data, data_type, scale_type, loss_formulation)
        Weighted_train_loss[key] = train_loss
        Weighted_test_loss[key] = test_loss
    
    X_Equivalent_train_loss = {'ep_x_eq':None, 'v_x_eq':None, 'sc_x_eq':None}
    X_Equivalent_test_loss = {'ep_x_eq':None, 'v_x_eq':None, 'sc_x_eq':None}
    Scaling = [
        ('ep_x_eq','EP_X_EQ'), 
        ('v_x_eq', 'V_X_EQ'), 
        ('sc_x_eq', 'SC_X_EQ')
    ]

    ## Train on x equivalent loss
    for key, scale_type in Scaling:
       loss_formulation = l.diffusion_loss_x
       train_loss, test_loss, _,_ = train_2D(train_data, test_data, data_type, scale_type, loss_formulation)
       X_Equivalent_train_loss[key] = train_loss
       X_Equivalent_test_loss[key] = test_loss

    _, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=False, sharey=False)

    # Plot only for training losses
    axs[0][0].plot(Weighted_test_loss['x'][1:], label='x-objective', color='blue')
    axs[0][0].set_title('X Objective(Test Loss)')
    axs[0][0].set_xlabel('Epochs')
    axs[0][0].set_ylabel('Loss')
    axs[0][0].legend()

    axs[0][1].plot(Weighted_test_loss['ep'][1:], label='ep-objective', color='blue')
    axs[0][1].plot(X_Equivalent_test_loss['ep_x_eq'][1:], label='ep_x_eq', color='red')
    axs[0][1].set_title('Ep vs Ep Equivalent X')
    axs[0][1].set_xlabel('Epochs')
    axs[0][1].set_ylabel('Loss')
    axs[0][1].legend()

    axs[1][0].plot(Weighted_test_loss['v'][1:], label='v-objective', color='blue')
    axs[1][0].plot(X_Equivalent_test_loss['v_x_eq'][1:], label='v_x_eq', color='red')
    axs[1][0].set_title('V vs V Equivalent X')
    axs[1][0].set_xlabel('Epochs')
    axs[1][0].set_ylabel('Loss')
    axs[1][0].legend()

    axs[1][1].plot(Weighted_test_loss['sc'][1:], label='sc-objective', color='blue')
    axs[1][1].plot(X_Equivalent_test_loss['sc_x_eq'][1:], label='sc_x_eq', color='red')
    axs[1][1].set_title('SC vs SC Equivalent X')
    axs[1][1].set_xlabel('Epochs')
    axs[1][1].set_ylabel('Loss')
    axs[1][1].legend()

    plt.tight_layout()
    utils.save_and_showfig(fname, show_figure = True)

    return Weighted_train_loss, Weighted_test_loss, X_Equivalent_train_loss, X_Equivalent_test_loss


def compare_ELBO_losses(data, data_type):
    """Function to compare ELBO objectives."""
    fname = f'results/{data_type}/Comparing_ELBO_loss'
    ELBO_train_loss = {}
    ELBO_test_loss = {}
    ELBO_objectives = {
        'x': l.diffusion_loss_x,
        'ep': l.diffusion_loss_epsilon,
        'v': l.diffusion_loss_v,
        'sc': l.diffusion_loss_score
    }

    train_data, test_data = split_train_test(data)
    
    for key, loss_func in ELBO_objectives.items():
        train_loss, test_loss, _ ,_= train_2D(train_data, test_data, data_type, 'ELBO', loss_func)
        ELBO_train_loss[key] = train_loss
        ELBO_test_loss[key] = test_loss

    _, axs = plt.subplots(1, 2, figsize=(8, 3))
    for ax, losses, title in zip(
        axs, [ELBO_train_loss, ELBO_test_loss], ["Training", "Testing"]
    ):
        for key, loss in losses.items():
            ax.plot(loss, label=f"{key.upper()}-ELBO")
        ax.set_title(f"{title} ELBO Loss for {data_type} data")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

    utils.save_and_showfig(fname, show_figure=True)


def plot_loss_by_t_per_epoch(data, epochs_to_plot, loss_formulation, data_type, scale_type):
    train_data, test_data = split_train_test(data)
    epochs_to_plot = [i for i in range(2,100,20)] + [99]
    _, _, _, loss_vs_time_per_epoch,_ = train_2D(train_data, test_data, data_type,scale_type, loss_formulation)
    for pos in epochs_to_plot:
        if pos < len(loss_vs_time_per_epoch):
            dictionary = loss_vs_time_per_epoch[pos]
            keys = list(dictionary.keys())
            values = list(dictionary.values())
            plt.plot(keys, values, label=f"Epoch {pos+1}")
        else:
            print(f"Position {pos} is out of range.")

    # Add labels, legend, and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Average Loss at Each Time Step")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def ELBO_losses_vs_time(data, data_type):
    """Function to compare ELBO loss wrt time steps"""
    epochs = config['epochs']
    fname = f'results/{data_type}/Comparing_ELBO_loss'
    ELBO_loss_by_t = {}
    ELBO_objectives = {
        'x': l.diffusion_loss_x,
        'ep': l.diffusion_loss_epsilon,
        'v': l.diffusion_loss_v,
        'sc': l.diffusion_loss_score
    }

    train_data, test_data = split_train_test(data)
    
    for key, loss_func in ELBO_objectives.items():
        _, _, _, loss_by_t_per_epoch = train_2D(train_data, test_data, data_type, 'ELBO', loss_func)
        ELBO_loss_by_t[key] = loss_by_t_per_epoch[epochs-1]
    plt.figure(figsize=(6, 4))
    for loss_type, losses in ELBO_loss_by_t.items():
        x_values = [x / 10 for x in losses.keys()]
        y_values = list(losses.values())
        plt.plot(x_values, y_values, label=f'Loss: {loss_type}')
    
    plt.xlabel('Time Step')
    plt.ylabel('ELBO Loss')
    plt.title('ELBO Loss vs. Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()


def loss_vs_time(loss_formulation, scale_type, data_dict):
    """ Function to get Loss vs Time for each data set"""
    epochs = config['epochs']
    loss_vs_time_dict = {}
    for key, val in data_dict.items():
        train_data, test_data = split_train_test(val)
        data_type = key
        _, _, _ , loss_vs_time= train_2D(train_data, test_data, data_type, scale_type, loss_formulation)
        loss_vs_time_dict[key] = loss_vs_time[epochs-1]
    
    plt.figure(figsize=(6, 4))
    for data_type, losses in loss_vs_time_dict.items():
        x_values = [x / 100 for x in losses.keys()]
        y_values = list(losses.values())
        plt.plot(x_values, y_values, label=f'Loss: {data_type}')

    plt.xlabel('Time Step')
    plt.ylabel(f'ELBO Loss after epoch {epochs}')
    plt.title('ELBO Loss vs. Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()


    

    


