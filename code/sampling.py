import torch
import numpy as np

def sampling_epsilon(model, data, device, num_steps, t_embed=False, clamp_x=False):
    """ Function to sample for noise prediction"""
    model.eval()
    with torch.no_grad():
        ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)            
        x_t = data.to(device)                                        

        for i in range(0, num_steps):
            t = torch.full((x_t.size(0),1), ts[i]).to(device)                         
            prev_t = torch.full((x_t.size(0),1), ts[i+1]).to(device)                  
            alpha_t = torch.cos((torch.pi/2)*t).view(-1, *([1] * (x_t.ndim - 1)))                         
            sigma_t = torch.sin((torch.pi/2)*t).view(-1, *([1] * (x_t.ndim - 1)))                             
            alpha_prev_t = torch.cos((torch.pi/2)*prev_t).view(-1, *([1] * (x_t.ndim - 1)))                  
            sigma_prev_t = torch.sin((torch.pi/2)*prev_t).view(-1, *([1] * (x_t.ndim - 1)))                  
            eta_t = sigma_prev_t / sigma_t * torch.sqrt(1 - alpha_t ** 2 / alpha_prev_t ** 2)
            noise_pred = model(x_t, t.squeeze(-1)) if t_embed else model(torch.cat([x_t, t], dim=1))
            min, max = (-1, 1) if clamp_x else (-float('inf'), float('inf'))                                      
            epsilon_t = torch.randn_like(x_t)                                                    
            root_term = torch.clamp(sigma_prev_t**2 - eta_t**2, min=0.0).sqrt()
            x_t = alpha_prev_t * torch.clamp(((x_t - sigma_t * noise_pred) / alpha_t), min=min,max=max) + root_term * noise_pred + eta_t * epsilon_t
    
    return x_t


def sampling_x(model, data, device, num_steps, t_embed=False, clamp_x=False):
    """ Function to sample for x prediction"""
    model.eval()
    with torch.no_grad():
        ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)           
        x_t = data.to(device)

        for i in range(0, num_steps):
            t = torch.full((x_t.size(0),1), ts[i]).to(device)                          
            prev_t = torch.full((x_t.size(0),1), ts[i+1]).to(device)                  
            alpha_t = torch.cos((torch.pi/2)*t).view(-1, *([1] * (x_t.ndim - 1)))                             
            sigma_t = torch.sin((torch.pi/2)*t).view(-1, *([1] * (x_t.ndim - 1)))                             
            alpha_prev_t = torch.cos((torch.pi/2)*prev_t).view(-1, *([1] * (x_t.ndim - 1)))                   
            sigma_prev_t = torch.sin((torch.pi/2)*prev_t).view(-1, *([1] * (x_t.ndim - 1)))                   
            eta_t = sigma_prev_t / sigma_t * torch.sqrt(1 - alpha_t ** 2 / alpha_prev_t ** 2)
            x0_pred = model(x_t, t.squeeze(-1)) if t_embed else model(torch.cat([x_t, t], dim=1))                          
            epsilon_t = torch.randn_like(x_t)                                                   
            root_term = torch.clamp(sigma_prev_t**2 - eta_t**2, min=0.0).sqrt()               
            x_t = alpha_prev_t * x0_pred + root_term * ((x_t - alpha_t*x0_pred)/ sigma_t) + eta_t * epsilon_t 
    
    return x_t


def sampling_v(model, data, device, num_steps, t_embed=False, clamp_x=False):
    """ Function to sample for v prediction"""
    model.eval()
    with torch.no_grad():
        ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1)                   
        x_t = data.to(device)                                      

        for i in range(0, num_steps):
            t = torch.full((x_t.size(0),1), ts[i]).to(device)                                                    
            prev_t = torch.full((x_t.size(0),1), ts[i+1]).to(device)                                             
            alpha_t = torch.cos(((torch.pi/2)*prev_t) - ((torch.pi/2)*t)).view(-1, *([1] * (x_t.ndim - 1)))                            
            sigma_t = torch.sin(((torch.pi/2)*prev_t) - ((torch.pi/2)*t)).view(-1, *([1] * (x_t.ndim - 1)))
            if t_embed:
                v_pred = model(x_t, t.squeeze(-1))
            else:                             
                v_pred = model(torch.cat([x_t, t], dim=1))
            x_t = alpha_t * x_t + sigma_t * v_pred          
    
    return x_t


def sampling_score(model, data, device, num_steps, t_embed=False, clamp_x=False):
    """ Function to sample for score model """
    model.eval()
    with torch.no_grad():
        x_t = data.to(device)
        t_values = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1).to(device) 
        sigma_t_values = torch.sin((torch.pi / 2) * t_values).view(-1, *([1] * (x_t.ndim - 1))).to(device)          
        
        for i in range(num_steps - 1):
            t = torch.full((x_t.size(0),1), t_values[i]).to(device)     
            sigma_t = sigma_t_values[i]
            eta_t = sigma_t ** 2
            if t_embed:
                score_pred = model(x_t, t.squeeze(-1))
            else:
                score_pred = model(torch.cat([x_t, t], dim=1))
            noise = torch.randn_like(x_t)
            x_t = x_t + eta_t * score_pred + torch.sqrt(2 * eta_t) * noise

    return x_t




