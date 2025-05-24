from lowmem_main import *
import heapq
import numpy as np
import scipy

def get_proposal_pmf(u, n):
  pmf = []
  K = np.arange(4)
  for i in reversed(range(n)):
    p = p_x(K*4**(i), u)
    p[p < 0.0] = 0.0
    p /= p.sum()
    p = p.astype(np.float64)

    pmf.append(p)
  return pmf


def p_x(idx, u):

  if u.ndim == 1: u = u.reshape( (-1,1) ) # make sure it's a column vector
  
  nQubits = int( np.log2( u.shape[0] ) )
  
  y = np.zeros((idx.shape[0]))
  k = 0
  for i in idx:
    v = u.copy()
    for ni,p in enumerate( reversed(int2lst(i, nQubits )) ):
    # for ni,p in enumerate( reversed(i) ):    
      v = PauliFcn_map[p]( v, 2**nQubits, ni)
    y[k] = np.real_if_close(np.vdot(u.copy(), v)**2/2**nQubits)
    k += 1
  
  return y
  
  
def cal_proba(idx, u):

  if u.ndim == 1: u = u.reshape( (-1,1) ) # make sure it's a column vector
  
  nQubits = int( np.log2( u.shape[0] ) )
  
  y = np.zeros((len(idx)))
  k = 0
  for i in idx:
    v = u.copy()
    for ni,p in enumerate( reversed(i) ):    
      v = PauliFcn_map[p]( v, 2**nQubits, ni)
    y[k] = np.real_if_close(np.vdot(u.copy(), v)**2/2**nQubits)
    k += 1
  
  return y  


def topK_products(sorted_vectors, sorted_keys, K):
    """
    Args:
        sorted_vectors: list of n arrays, each sorted descending
        sorted_keys: list of n arrays of original keys matching sorted_vectors
        K: top-K products to find

    Returns:
        topK_products: list of top-K products
        topK_keys: list of top-K index choices (in original array index space)
    """

    n = len(sorted_vectors)

    # Step 1: Initialize
    init_indices = [0] * n
    init_product = np.prod([sorted_vectors[i][0] for i in range(n)])

    heap = [(-init_product, init_indices)]  # max heap (negate products)
    seen = {tuple(init_indices)}

    topK_products = []
    topK_keys = []

    while len(topK_products) < K:
        neg_prod, indices = heapq.heappop(heap)

        # Map back to original keys
        original_keys = [sorted_keys[i][indices[i]] for i in range(n)]

        topK_products.append(-neg_prod)
        topK_keys.append(original_keys)

        # Expand neighbors
        for i in range(n):
            if indices[i] + 1 < len(sorted_vectors[i]):
                new_indices = indices.copy()
                new_indices[i] += 1
                new_key = tuple(new_indices)

                if new_key not in seen:
                    new_product = np.prod([sorted_vectors[j][new_indices[j]] for j in range(n)])
                    heapq.heappush(heap, (-new_product, new_indices))
                    seen.add(new_key)

    return topK_keys
    
    
def sample_from_arrays(values_list, probs_list, n_samples):
    """
    Args:
        values_list: list of n arrays (each containing [0,1,2,3])
        probs_list: list of n arrays (each giving probability distribution over values)

    Returns:
        sampled_values: list of n sampled values (one from each array)
    """

    samples = []
    for values, probs in zip(values_list, probs_list):
        sample = np.random.choice(values, p=probs, size =  n_samples)
        samples.append(sample)
        
    
    samples = np.array(samples)  # Shape becomes (5, 100)

    # Now transpose to (100, 5)
    samples = samples.T  
    samples = np.unique(samples, axis=0)
    mask = ~(np.all(samples == 0, axis=1))

    # Apply mask
    samples = samples[mask]

    return samples.tolist()
    
    
    
    
    
    