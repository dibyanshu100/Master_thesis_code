import numpy as np
from scipy.stats import skew, kurtosis

## Evaluation metrics for 2D generated samples

# Compute Mean Distance 
def compute_mean_distance(real_data, gen_data):
    real_mean = np.mean(real_data, axis=0)
    gen_mean = np.mean(gen_data, axis=0)
    return np.linalg.norm(real_mean - gen_mean)

# Compute Covariance Matrix Distance
def compute_covariance_distance(real_data, gen_data):
    real_cov = np.cov(real_data, rowvar=False)
    gen_cov = np.cov(gen_data, rowvar=False)
    return np.linalg.norm(real_cov - gen_cov, 'fro')

# Aggregate All Metrics
def compute_moment_metrics(real_data, gen_data):
    metrics = {
        'mean_distance': compute_mean_distance(real_data, gen_data),
        'covariance_distance': compute_covariance_distance(real_data, gen_data)
    }
    return metrics

