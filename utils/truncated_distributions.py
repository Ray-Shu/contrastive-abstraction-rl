import numpy as np 

def truncated_normal(len, mu, sigma): 
    """
    Creates an array of probabilities based on a Gaussian distribution. 

    Args: 
        len: Length of the probability array. 
        mu: The index to center the Gaussian distribution on. 
        sigma: The standard deviation. 
    """
    x = np.linspace(start=0, stop=len-1, num=len) 
    mu = mu 
    sigma = sigma 
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    pdf[mu] = 0 # set the probability of choosing the anchor state to zero. 

    probabilities = pdf / np.sum(pdf)

    return probabilities 

def truncated_laplace(len, mu, b): 
    """
    Creates an array of probabilities based on a Laplacian distribution. 

    Args: 
        len: Length of the probability array. 
        mu: The index to center the Laplace distribution on. 
        b: The scale parameter. 
    """
    x = np.linspace(start=0, stop=len-1, num=len) 
    mu = mu 
    b = b 
    pdf = np.exp(-np.abs(x - mu) / b) 

    pdf[mu] = 0 # set the probability of choosing the anchor state to zero. 

    probabilities = pdf / np.sum(pdf)

    return probabilities

def truncated_exponential(len, anchor_state_index, rate): 
    """
    Creates an array of probabilities based on an exponential distribution. 

    Args: 
        len: Length of the probability array. 
        rate: The lambda parameter. 
    """
    x = np.linspace(start=0, stop=len-1, num=len) 
    lambda_ = rate 
    pdf = np.exp(-lambda_ * x)
    
    pdf[anchor_state_index] = 0

    probabilities = pdf / np.sum(pdf)

    return probabilities
