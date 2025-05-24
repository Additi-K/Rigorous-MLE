import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from time import time
from tqdm import tqdm
import numpy as np
import scipy
from scipy.linalg import norm
trace = lambda rho : np.real_if_close(np.trace(rho))


from Basis.Basic_Function import get_default_device
from Basis.Basis_State import Mea_basis, State
from Basis.Loss_Function import MLE_loss
from povm_subset import *
# from models.others.lbfgs_bm import lowmem_lbfgs_nn, lowmem_lbfgs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#######################################################
##low memory implementation for probability calculation
#######################################################

#######################################################
## numpy implementation
#######################################################

def int2lst(j, nQubits ):
  """ input: 0 <= j < 4^nQubits,
  where j represents one of the measurements.
  We'll take the 4-nary representation of j
  and return it as a list of length nQubits
  Ex: j = 19 and nQubits = 4 returns [0, 0, 1, 0, 3]
  """
  pad = nQubits-np.floor(np.log(np.maximum(j,.3))/np.log(4))-1
  lst = np.base_repr( j, base=4, padding=int(pad) )

  return [int(i) for i in lst ]
  # return list(map(int, lst))

def optimized_orderX(n, i):
  base_pattern = np.concatenate([
      np.full(2**i, 2**i),
      np.full(2**i, -2**i)
  ])  # Create base pattern

  order = np.tile(base_pattern, 2**(n - (i + 1)))  # Efficient tiling
  return order  # No need for extra multiplications


def optimized_scaleZ(n, i):
  base_pattern = np.concatenate([
      np.full(2**i, 1),
      np.full(2**i, -1)
  ])  # Create base pattern

  scale = np.tile(base_pattern, 2**(n - (i + 1)))  # Efficient tiling
  return scale.reshape(-1, 1)  # No need for extra multiplications


def testX(u, N, i):
  n = int(np.log2(N))

  order = optimized_orderX(n, i)

  idx = np.arange(N)

  return u[idx+order]

def testZ(u, N, i):
  n = int(np.log2(N))

  scale = optimized_scaleZ(n, i)
  # u *= scale[:, np.newaxis]
  u *= scale

  return u


def testY(u, N, i):
  n = int(np.log2(N))

  # scale = [-1j]*2**i + [1j]*2**i
  # scale = np.array(scale*(2**(n-(i+1))))
  # order = [2**i]*2**i + [-2**i]*2**i
  # order = np.array(order*(2**(n-(i+1))))

  scale = -1j*optimized_scaleZ(n, i)
  order = optimized_orderX(n, i)

  idx = np.arange(N)

  # return scale[:, np.newaxis]* u[idx+order]
  return scale* u[idx+order]

def testI(u, N, i):
  return u

PauliFcn_map = {0:testI, 1:testZ, 2:testX, 3:testY}

def lowmemAu(u, meas):

  """ March 2025, write code without using tensor reshapes...
  This is the building block for other codes, e.g., ones that do tr( A@u@u.T ) = vdot(u,A@u)
      or ones that build this up for a gradient

  To extend this, will need to:
      - allow rank r>1  (we may want to transpose u so that it's r x n not n x r,
                          because we'd want the new code to work on the "fast" dimension)
      - loop over multiple measurements
      - accumulate into a gradient (e.g., we need the frequencies, and the "flag" of +1 or -1 to convert from Pauli to POVM)

  We might also need an adjoint operator, unless we do backpropagation...

  """
  m = len(meas)
  
  y = np.zeros((2*m, ))
  
  nQubits = int( np.log2( u.shape[0] ) )

  srt = time()

  tr =  np.real_if_close( np.vdot(u, u) )

  
#   for i in range(meas.shape[0]):
  for i in range(len(meas)):

    v = u.copy()
    # for ni,p in enumerate( reversed(int2lst(meas[i], nQubits )) ):
    for ni,p in enumerate( reversed(meas[i]) ):
        v = PauliFcn_map[p]( v, 2**nQubits, ni)
      
    y[i] = np.real_if_close( np.vdot(u, v) )  

  temp = y[0:m]
  y[0:m] = 0.5*(tr+temp)
  y[m:] = 0.5*(tr-temp)
  
  return y
  

def lowmemAu_all(u, meas):

  """ March 2025, write code without using tensor reshapes...
  This is the building block for other codes, e.g., ones that do tr( A@u@u.T ) = vdot(u,A@u)
      or ones that build this up for a gradient

  To extend this, will need to:
      - allow rank r>1  (we may want to transpose u so that it's r x n not n x r,
                          because we'd want the new code to work on the "fast" dimension)
      - loop over multiple measurements
      - accumulate into a gradient (e.g., we need the frequencies, and the "flag" of +1 or -1 to convert from Pauli to POVM)

  We might also need an adjoint operator, unless we do backpropagation...

  """
  m = len(meas)
  # if torch.is_tensor(u):
  #     u_np = u.detach().cpu().numpy()
  
  y = np.zeros((m, 1))
  
  nQubits = int( np.log2( u.shape[0] ) )

  srt = time()
  for i in range(meas.shape[0]):
      v = u.copy()
      for ni,p in enumerate( reversed(int2lst(meas[i], nQubits )) ):
          v = PauliFcn_map[p]( v, 2**nQubits, ni)
      
      y[i] = np.real_if_close( np.vdot(u, v) )  
  
  print('time:', time()-srt)

  return y  

class TimeExceededException(Exception):
    """Custom exception to stop optimization when time limit is exceeded."""
    pass

class TimingCallback:
  def __init__(self, fun, fidelity, reshape, n_qubits, m, lmbda, timeLimit, rho_star):
      self.lmbda = lmbda
      self.fidelity = fidelity
      self.fun = fun
      self.reshape = reshape
      self.timeLimit = timeLimit
      self.cnt = 0
      self.t2 = time()
      self.rho_star = rho_star
      self.result_save = {'n_qubits':n_qubits,
                          'm':m,
                          'time_all': [],
                          'epoch': [],
                          'Fq': [], 
                          'loss': [],
                          'error':[],
                          'fmin': 0.0}
      self.result_save['fmin'] =  self.fun(self.rho_star)-  0.5*self.lmbda*np.linalg.norm(self.rho_star)**2      


  def __call__(self, intermediate_result):
      self.t1 = time()
      u = intermediate_result.copy()
      self.result_save['u_est'] = u
    
      # update stats
      self.result_save['epoch'].append(self.cnt)
      self.cnt += 1
    
      fval = self.fun(u) - 0.5*self.lmbda*np.linalg.norm(u)**2
      self.result_save['loss'].append(fval)
      error = np.linalg.norm( fval-self.result_save['fmin'] )**2
      self.result_save['error'].append(error )
      
      fid = self.fidelity(u)
      self.result_save['Fq'].append(fid)
    
      t = self.t1-self.t2
      self.t2 = time()
      self.result_save['time_all'].append(t) 
      
      if self.cnt%10 == 0:
        print("LBFGS_BM error {:.8f} | Fq {:.8f} | time {:.5f}".format(error, fid, np.sum(self.result_save['time_all'])))
        # Check if total elapsed time exceeds the limit
        if sum(self.result_save['time_all']) > self.timeLimit:
          print(f"Time limit exceeded: {self.result_save['time_all'][-1]:.2f}s > {self.timeLimit:.2f}s")
          raise TimeExceededException("Time limit exceeded, stopping optimization")



class LBFGS_numpy():
  def __init__(self, opt):
    self.rho_star = opt.rho_star
    self.f = opt.data
    self.n_qubits = opt.n_qubits
    self.rank = opt.rank
    self.povm = opt.subset
    self.m = len(self.povm)
    self.timeLimit = 3600*4
    self.lmbda = 2*np.sum(self.f)
    self.callback = TimingCallback(self.forward, self.fidelity, self.reshape, self.n_qubits, self.m, self.lmbda, self.timeLimit, self.rho_star)

    # Initialize u
    dim = 2*self.rank*(2)**self.n_qubits
    self.u0 = np.random.randn((dim))
    self.u0 = self.u0/ np.linalg.norm(self.u0)
    
    temp = self.forward(self.rho_star)
    print('initial error:', np.abs(temp- self.forward(self.u0))/np.abs(temp))
  
  def forward(self, u):
    # return function value
    if u.shape != 2**self.n_qubits:
        u = self.reshape(u).astype(np.complex64)
    p_out = lowmemAu(u, self.povm)
    val = -np.dot(self.f, np.log(p_out)) + 0.5* self.lmbda*np.linalg.norm(u)**2
    return val


  def gradient(self, u):
    # m = self.povm.shape[0]
    m = len(self.povm)
    u = self.reshape(u).astype(np.complex64)

    grad = np.zeros(u.shape, dtype = np.complex64)
    tr =  np.real_if_close( np.vdot(u, u) )
  
    for i in range(m):
      v = u.copy()
    #   for ni,p in enumerate( reversed(int2lst(self.povm[i], self.n_qubits )) ):
      for ni,p in enumerate( reversed(self.povm[i]) ):      
        v = PauliFcn_map[p]( v, 2**self.n_qubits, ni)
      temp = np.vdot(u, v)  
      grad +=  1.0*(self.f[i]*(u + v)/(tr + temp) + self.f[i+m]*(u - v)/(tr - temp))
    
    grad *= -2
    grad+= self.lmbda*u
    grad = np.vstack((np.real(grad).reshape((self.rank*2**self.n_qubits,1), order='C') , np.imag(grad).reshape((self.rank*2**self.n_qubits,1), order='C')))
    grad = grad.astype(np.float64)
     
    return grad

  def optimize(self):
    """Run L-BFGS-B to minimize the objective starting from x0."""
    try:
      result = scipy.optimize.minimize(
          fun=self.forward,
          x0=self.u0,
          jac=self.gradient,             # use analytic gradient
          method='L-BFGS-B',
          callback=self.callback,
          options={
            'disp': False,
            'ftol': 1e-9,
            'gtol': 1e-9,
            'iprint': -1,
            'maxiter': 1000,
            'maxfun': 10000,
            'maxcor': 2
              }
      )

    except TimeExceededException as e:
        u_est = self.callback.result_save['u_est']
        X = cal_proba(self.povm, self.reshape(u_est))/cal_proba(self.povm, self.rho_star)
        Y =np.mean(X)
        return self.callback.result_save, Y

    u_est = self.callback.result_save['u_est']
    X = cal_proba(self.povm, self.reshape(u_est))/cal_proba(self.povm, self.rho_star)
    Y =np.mean(X)
    return self.callback.result_save, Y

    
  def reshape(self, u):
      
    if u.shape[0] == 2**self.n_qubits:
        return u
    else:    
        idx = int(u.shape[0]/2)
        reshaped_u = u[0:idx].reshape((2**self.n_qubits, self.rank), order = 'C') + 1j*u[idx:].reshape((2**self.n_qubits, self.rank), order='C')
        return reshaped_u

  def fidelity(self, u):
    u = self.reshape(u).astype(np.complex64)
    return np.linalg.norm(np.vdot(self.rho_star, u))**2 
    
    
'''
def Dataset_sample_lowmem_dfe(n_qubits, rho_star):
    
  pmf_list = get_proposal_pmf(rho_star, n_qubits)
  # Sort each mini-array descending, and keep original indices
  sorted_vectors = []
  sorted_keys = []
    
  for v in pmf_list:
    idx = np.argsort(-v)
    sorted_vectors.append(v[idx])
    sorted_keys.append(idx)
    
    
  # number of povms to take
  epsilon = 0.03
  delta = 0.10                                                  
  K = min(int(np.ceil(np.log(1 / delta) / (epsilon ** 2))), int(4**n_qubits))
  subset = topK_products(sorted_vectors, sorted_keys, K)
  subset.pop(0)
  K = len(subset)
  
  proba = lowmemAu(rho_star, subset)
  proba = proba[0:K]
  
  out1 = np.random.binomial( 100, np.real_if_close(proba))
  out2 = 100 - out1
  out1 = out1/100
  out2 = out2/100
  
#   meas = []
#   for i in subset:
#     meas.append(base4_to_base10(i))
    

  return subset , np.concatenate((out1, out2))
  
'''

def Dataset_sample_lowmem_dfe(n_qubits, rho_star, lmbda):
    
  pmf_list = get_proposal_pmf(rho_star, n_qubits)
  values_list = [np.array([0,1,2,3]) for _ in range(n_qubits)]
    
    
  # number of povms to take
  epsilon = 0.03
  delta = 0.10                                                  
  K = min(int(np.ceil(np.log(1 / delta) / (epsilon ** 2))), int(4**n_qubits))
  
  subset = sample_from_arrays(values_list,  pmf_list, K)
#   m = np.zeros((len(subset)), dtype = int)
  
  weights = cal_proba(subset, rho_star)
  weights = weights.ravel()

#   m = np.ceil(2*np.log(2/delta)/ (2**n_qubits*weights*K*epsilon**2)).astype(int)
  m = 100
  K = len(subset)
  proba = lowmemAu(rho_star, subset)
  proba = proba[0:K]
  proba = (1-lmbda)*proba + lmbda*0.5
  
  out1 = np.random.binomial( m, np.real_if_close(proba))
  out2 = m - out1
  out1 = out1/m 
  out2 = out2/m 
  
#   meas = []
#   for i in subset:
#     meas.append(base4_to_base10(i))
    

  return subset , np.concatenate((out1, out2))
  
  

def Dataset_sample_lowmem_random(n_qubits, rho_star):
  all = np.arange(0, 4**n_qubits) 
  pmf = lowmemAu_all(rho_star, all)
#   r_path = '/scratch/alpine/kuad8709/QST/data/pauli/'

#   np.save(r_path +  'lowmem' + '_' + 'true_' + str(opt.n_qubits)  + '.npy', pmf)
  weights = pmf**2/2**n_qubits

  # number of povms to take
  epsilon = 0.03
  delta = 0.10                                                  
  l = min(int(np.ceil(np.log(1 / delta) / (epsilon ** 2))), len(weights) - 1)
  meas = np.argsort(weights[1:], axis=0)[-l:]
  meas = np.sort(meas)

  pmf = 0.5*(pmf[0] + pmf[meas+1]).flatten()

  out1 = np.random.binomial( 100, np.real_if_close(pmf))
  out2 = 100 - out1
  out1 = out1/100
  out2 = out2/100
  
  subset = []
  for j in (meas+1):
      subset.append(int2lst(int(j), n_qubits ))

  return subset, np.concatenate((out1, out2))
    
   

def train(opt):

  state_star = get_u_product(opt.n_qubits)
  subset, data = Dataset_sample_lowmem_dfe(opt.n_qubits, state_star, 0.1)


  opt.rho_star = state_star
  opt.data = data
  opt.subset = subset
  print('setup complete')
  optimizer = LBFGS_numpy(opt)
  #res = optimizer.optimize()
  res, Y = optimizer.optimize()
  
  print('DFE estimate:', Y)
  print('low-memory algorithm estimate:', res['Fq'][-1])

  return res
 
def get_u_product(n):
  u = np.random.rand(2,1)
  for i in range(n-1):
    u = np.kron(u, np.random.rand(2,1))

  u /= np.linalg.norm(u)

  return u
  
def base4_to_base10(idx_list):
  n = len(idx_list)
  powers = 4 ** np.arange(n-1, -1, -1)
  
  return int(np.dot(idx_list, powers))  
  
  

  

if __name__ == '__main__':
  # ----------parameters----------
  print('-'*20+'set parser'+'-'*20)
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_qubits", type=int, default=18, help="number of qubits")
  parser.add_argument("--rank", type=float, default=1, help="rank of mixed state")
  opt = parser.parse_args()

  r_path = '/scratch/alpine/kuad8709/QST/data/pauli/'
  result = train(opt)

  np.save(r_path +  'lowmem' + '_' + str(opt.n_qubits)  + '_m_100' + '_depol_0.1' + '.npy', result)
