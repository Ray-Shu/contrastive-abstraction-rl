import numpy as np 
import faiss

def remove_dupes(x, k=1000, threshold=0.99): 
    """
    Removes duplicate or similar vectors in the matrix "x". 

    Args: 
        x: The input matrix, size [N, d], with N vectors of size d. 
        k: The number of vectors that are used in similarity calculations per vector. 
        threshold: The cosine similarity threshold to find unique vectors. 
            The lower the threshold, the more unique the vectors will be. 
    
    Returns: 
        Returns a "mask", that holds the information of which vectors are too similar to other vectors. 
        Then, to obtain the unique variant, do x[mask].
    """
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)

    N = x.shape[0]
    mask = np.ones(N, dtype=bool)

    for i in range(N):
        if not mask[i]:
            continue
        D, I = index.search(x[i:i+1], k+1)
        neighbors = I[0, 1:]
        sims = D[0, 1:]
        for n, sim in zip(neighbors, sims):
            if sim > threshold:
                mask[n] = False

    return mask