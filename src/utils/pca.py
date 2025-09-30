import torch
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

def process_states(states, model) -> list: 
    """
    Takes in raw states and returns its corresponding 2D PCA embedding. 
    Args: 
        states: The raw states, of size [N, d]
        model: the model to obtain learned representations (z)
    Returns: 
        A list of PCA embeddings from the 4 models in this order: 
        [pca_laplace, pca_gauss, pca_expo, pca_uniform]
    """
    # Convert subsampled states to latent represntation 
    new_states = torch.as_tensor(states, dtype=torch.float32)

    with torch.no_grad():
        z = model(new_states)
        
    # Normalize representation embeddings 
    scaler = StandardScaler()
    norm_z = scaler.fit_transform(z)

    # PCA 
    pca = PCA(n_components=2)
    pca_z = pca.fit_transform(norm_z)

    return {"pca-reps": pca_z, 
            "pca-models": pca,
            "scalars": scaler}

def pca_transform(states, pca_dict, model, has_representation = False) -> list[list]:
    """
    Transforms states into their pca transformation given an existing pca model to do so. 

    Args: 
        states: The input states to be PCA-ed
        pca_dict: The PCA information (models, scalars)
        model: The model to use, which must correspond to the type of distribution.
        has_representation: A bool to decide if the input are raw states, or a representation.  
    
    Returns: 
        An [N, 2] list, which is the states being PCA-ed. 
    """

    pca_model = pca_dict["pca-models"]
    scalar = pca_dict["scalars"]

    if has_representation == False: 
        with torch.no_grad(): 
            z = model(torch.as_tensor(states, dtype=torch.float32))
    else: 
        z = states

    scaled_z = (z - torch.as_tensor(scalar.mean_, dtype=z.dtype)) / torch.as_tensor(scalar.scale_, dtype=z.dtype)

    mean = torch.as_tensor(pca_model.mean_, dtype=torch.float32)
    centered_z = scaled_z - mean

    W = torch.as_tensor(pca_model.components_, dtype=torch.float32)

    y = torch.matmul(centered_z, W.T)

    return y