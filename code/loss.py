import torch
import torch.nn.functional as F



def diffusion_loss_epsilon(x, model, device, method, scale_type, t_embedded = False, loss_by_t=None):
    """ Function to calculate loss for epsilon parameterization """
    t = torch.rand((x.size(0),1)).to(device)  
    alpha_t = torch.cos((torch.pi/2)*t)         
    sigma_t = torch.sin((torch.pi/2)*t)

    # Expand alpha_t and sigma_t to match the shape of x
    alpha_t = alpha_t.view(-1, *([1] * (x.ndim - 1)))
    sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))

    noise = torch.randn_like(x).to(device)                
    x_t = alpha_t * x + sigma_t * noise

    noise_pred = model(x_t, t.squeeze(-1)) if t_embedded else model(torch.cat([x_t, t], dim=1))
    #scaling_coeff = 1 / (sigma_t * alpha_t + 1e-2) if scale_type == 'ELBO' else 1
    if scale_type == 'ELBO':
        scaling_coeff = 1 / (sigma_t * alpha_t + 1e-2)
    elif scale_type == 'Rescaled':
        scaling_coeff = (sigma_t**2) / ((alpha_t**2) + 1e-2)
    else:
        scaling_coeff = 1
    # Calculate per-sample MSE loss
    sample_loss = scaling_coeff * F.mse_loss(noise, noise_pred, reduction='none')  
    sample_loss = sample_loss.view(sample_loss.size(0), -1).mean(dim=1)

    # Track loss by time step while training
    if method == "train":
        time_steps = (t * 100).int().squeeze()             
        for i in range(x.size(0)):                          
            step = time_steps[i].item()                     
            loss_by_t[step].append(sample_loss[i].item())   
            
    return sample_loss.mean()


def diffusion_loss_x(x, model, device, method, scale_type, t_embedded = False, loss_by_t=None):
    """ Function to calculate loss for x parameterization """
    t = torch.rand((x.size(0),1)).to(device)             
    alpha_t = torch.cos((torch.pi/2)*t)         
    sigma_t = torch.sin((torch.pi/2)*t)

    # Expand alpha_t and sigma_t to match the shape of x
    alpha_t = alpha_t.view(-1, *([1] * (x.ndim - 1)))
    sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))

    noise = torch.randn_like(x).to(device)                
    x_t = alpha_t * x + sigma_t * noise
    x0_pred = model(x_t, t.squeeze(-1)) if t_embedded else model(torch.cat([x_t, t], dim=1))
    if scale_type == 'ELBO':
        scaling_coeff = alpha_t/((sigma_t**3)+1e-2) 
    elif scale_type == 'EP_X_EQ':
        scaling_coeff = (alpha_t**2)/((sigma_t**2)+1e-2)
    elif scale_type == 'V_X_EQ':
        scaling_coeff = ((alpha_t**2 + sigma_t**2)/(sigma_t**2+1e-2))
    elif scale_type == 'SC_X_EQ':
        scaling_coeff = (alpha_t**2)/((sigma_t**4)+1e-2)
    else:
        scaling_coeff = 1

    # Calculate per-sample MSE loss
    sample_loss = scaling_coeff * F.mse_loss(x, x0_pred, reduction='none')
    sample_loss = sample_loss.view(sample_loss.size(0), -1).mean(dim=1)

    # Track loss by time step while training
    if method == "train":
        time_steps = (t * 100).int().squeeze()             
        for i in range(x.size(0)):                          
            step = time_steps[i].item()                     
            loss_by_t[step].append(sample_loss[i].item())   
            
    return sample_loss.mean()


def diffusion_loss_v(x, model, device, method, scale_type, t_embedded = False, loss_by_t=None):
    """ Function to calculate loss for v parameterization """
    t = torch.rand((x.size(0), 1)).to(device)
    alpha_t = torch.cos((torch.pi / 2) * t) 
    sigma_t = torch.sin((torch.pi / 2) * t)

    # Expand alpha_t and sigma_t to match the shape of x
    alpha_t = alpha_t.view(-1, *([1] * (x.ndim - 1)))
    sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))

    noise = torch.randn_like(x).to(device)
    x_t = alpha_t * x + sigma_t * noise
    v_t = alpha_t * noise  - sigma_t * x
    v_pred = model(x_t, t.squeeze(-1)) if t_embedded else model(torch.cat([x_t, t], dim=1))
    #scaling_coeff = alpha_t/((sigma_t*(alpha_t**2 + sigma_t**2))+1e-2) if scale_type == 'ELBO' else 1

    if scale_type == 'ELBO':
        scaling_coeff = alpha_t/((sigma_t*(alpha_t**2 + sigma_t**2))+1e-2)
    elif scale_type == 'Rescaled':
        scaling_coeff = (sigma_t**2)/((alpha_t**2 + sigma_t**2)+1e-2)
    else:
        scaling_coeff = 1

    # Calculate per-sample MSE loss
    sample_loss = scaling_coeff * F.mse_loss(v_t, v_pred, reduction='none')
    sample_loss = sample_loss.view(sample_loss.size(0), -1).mean(dim=1)

    # Track loss by time step while training
    if method == "train":
        time_steps = (t * 100).int().squeeze()
        for i in range(x.size(0)):
            step = time_steps[i].item()
            loss_by_t[step].append(sample_loss[i].item())

    return sample_loss.mean()


def diffusion_loss_score(x, model, device, method, scale_type, t_embedded = False, loss_by_t=None):
    """ Function to calculate loss for v parameterization """
    t = torch.rand((x.size(0), 1)).to(device)
    alpha_t = torch.cos((torch.pi / 2) * t) 
    sigma_t = torch.sin((torch.pi / 2) * t)

    # Expand alpha_t and sigma_t to match the shape of x
    alpha_t = alpha_t.view(-1, *([1] * (x.ndim - 1)))
    sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))
    noise = torch.randn_like(x).to(device)
    x_t = alpha_t * x + sigma_t * noise
    score_pred = model(x_t, t.squeeze(-1)) if t_embedded else model(torch.cat([x_t, t], dim=1))
    target_score = -(x_t - alpha_t * x) / ((sigma_t**2) + 1e-2)
    #scaling_coeff = sigma_t/(alpha_t+1e-2) if scale_type == 'ELBO' else 1
    #scaling_coeff = (sigma_t**2) if scale_type == 'ELBO' else 1

    if scale_type == 'ELBO':
        scaling_coeff = (sigma_t**2)
    elif scale_type == 'Rescaled':
        scaling_coeff = (sigma_t**4)/((alpha_t**2) + 1e-2)
    else:
        scaling_coeff=1

    # Calculate per-sample MSE loss
    sample_loss = scaling_coeff * F.mse_loss(score_pred, target_score, reduction='none')
    sample_loss = sample_loss.view(sample_loss.size(0), -1).mean(dim=1)

    # Track loss by time step while training
    if method == "train":
        time_steps = (t * 100).int().squeeze()
        for i in range(x.size(0)):
            step = time_steps[i].item()
            loss_by_t[step].append(sample_loss[i].item())
    
    return sample_loss.mean()


def equivalent_x_loss(x, model, device, loss_type):

    t = torch.rand((x.size(0),1)).to(device)             
    alpha_t = torch.cos((torch.pi/2)*t)         
    sigma_t = torch.sin((torch.pi/2)*t)        
    noise = torch.randn_like(x).to(device)                
    x_t = alpha_t * x + sigma_t * noise
    input = torch.cat([x_t,t], dim=1)          
    x0_pred = model(input)

    if loss_type == 'diffusion_loss_x':       
        scaling_coeff = 1
    if loss_type == 'diffusion_loss_epsilon':
        scaling_coeff = (alpha_t**2)/((sigma_t**2)+1e-2)
    if loss_type == 'diffusion_loss_v':
        scaling_coeff = ((alpha_t**2 + sigma_t**2)/(sigma_t+1e-2))**2
    if loss_type == 'diffusion_loss_score':
        scaling_coeff = 1/((sigma_t**4)+1e-2)     

    # Calculate per-sample MSE loss
    sample_loss =  scaling_coeff * F.mse_loss(x, x0_pred, reduction='none')                            
            
    return sample_loss.mean()



