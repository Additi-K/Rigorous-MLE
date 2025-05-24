#torch implementation of lbfgs-bm

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from tqdm import tqdm

sys.path.append('../..')

from models.UGD.rprop import Rprop
from models.UGD.cg_optim import cg
from Basis.Basis_State import Mea_basis, State
from evaluation.Fidelity import Fid
from Basis.Loss_Function import MLE_loss, LS_loss, CF_loss
from Basis.Basic_Function import qmt_torch, get_default_device, proj_spectrahedron_torch, qmt_matrix_torch

class lbfgs_nn(nn.Module):

  def __init__(self, na_state, 
                 n_qubits,
                 P_idxs,
                 M, rank):
    super().__init__()
                   
    self.N = n_qubits
    self.P_idxs = P_idxs
    self.M = M
    self.device = M.device 
    self.rank = rank#np.maximum(1, int(2**n_qubits/4))           

    d = 2**n_qubits
    params = torch.randn((2, d, self.rank), requires_grad=True).to(torch.float32)
    self.params = nn.Parameter(params)

  def forward(self):
    self.rho = self.Rho()
    P_out = self.Measure_rho()  # perfect measurement
    return P_out

  def Rho(self):
    U = torch.complex(self.params[0,:,:], self.params[1,:,:])
    rho = torch.matmul(U, U.T.conj())
    rho = rho / torch.trace(rho)
    return rho

  def Measure_rho(self):
    """Born's Rule"""
    self.rho = self.rho.to(torch.complex64)
    P_all = qmt_torch(self.rho, [self.M] * self.N)

    P_real = P_all[self.P_idxs]
    return P_real

class lbfgs():
  def __init__(self, na_state, generator, P_star, learning_rate=0.01):
      self.generator = generator
      self.P_star = P_star
      self.criterion = MLE_loss

      self.optim = optim.LBFGS(self.generator.parameters(), lr=0.1, max_iter=1000, 
                               tolerance_grad=1e-07, tolerance_change=1e-09, 
                               history_size=10, line_search_fn=None)
      
      self.overhead_t = 0
      self.epoch = 0
      self.time_all = 0 

  def track_parameters(self, loss, fid, result_save):
      """Callback to store parameter updates (excluding computation time)."""

      start_overhead = perf_counter()  # Start timing overhead
      self.generator.eval()

      with torch.no_grad():
          rho = self.generator.rho
          rho /= torch.trace(rho)
          penalty = 0.5 * 2 * torch.sum(self.P_star) * torch.norm(self.generator.params, p=2) ** 2

          Fq = fid.Fidelity(rho)

          result_save['epoch'].append(self.epoch)
          result_save['Fq'].append(Fq)
          result_save['loss'].append(loss.item() - penalty)
          self.epoch += 1

      self.overhead_t = perf_counter() - start_overhead  # ✅ Correct overhead timing

  def train(self, epochs, fid, result_save):
      """Net training."""
      pbar = tqdm(range(1), mininterval=0.01)
      epoch = 0

      for _ in pbar:
          epoch += 1
          

          self.generator.train()

          def closure():
              self.generator.train()
              time_b = perf_counter()
              self.optim.zero_grad()
              P_out = self.generator()
              loss = self.criterion(P_out, self.P_star)
              loss += 0.5 * 2 * torch.sum(self.P_star) * torch.norm(self.generator.params, p=2) ** 2
              
              assert not torch.isnan(loss), "Loss is NaN" 
              loss.backward()
              self.track_parameters(loss, fid, result_save)
              # Update tracking (exclude overhead from time_all)
              raw_t = perf_counter()
              self.time_all += raw_t - time_b - self.overhead_t
              result_save['time'].append(self.time_all)

              return loss

          self.optim.step(closure)

      # Print tracked updates
      for i, (f, l, t) in enumerate(zip(result_save['Fq'], result_save['loss'], result_save['time'])):
          print("LBFGS_BM loss {:.10f} | Fq {:.8f} | time {:.5f}".format(l, f, t))

      pbar.close()

  # def train(self, epochs, fid, result_save):
  #   """Net training"""
  #   # self.sche = optim.lr_scheduler.StepLR(self.optim, step_size=1500, gamma=0.2)

  #   pbar = tqdm(range(epochs), mininterval=0.01)
  #   epoch = 0
  #   time_all = 0
  #   for i in pbar:
  #       epoch += 1
  #       time_b = perf_counter()

  #       self.generator.train()

  #       def closure():
  #           self.optim.zero_grad()
  #           data = self.P_star
  #           P_out = self.generator()
  #           loss = self.criterion(P_out, data)
  #           loss += 0.5*2*torch.sum(self.P_star)*torch.norm(self.generator.params,  p=2)**2
  #           assert torch.isnan(loss) == 0, print('loss is nan', loss)
  #           loss.backward()
  #           return loss

  #       self.optim.step(closure)
  #       # self.sche.step()

  #       time_e = perf_counter()
  #       time_all += time_e - time_b

  #       # show and save
  #       if epoch % 10 == 0 or epoch == 1:
  #           loss = closure().item()
  #           self.generator.eval()
  #           with torch.no_grad():
  #               rho = self.generator.rho
  #               rho /= torch.trace(rho)
  #               penalty = 0.5*2*torch.sum(self.P_star)*torch.norm(self.generator.params,  p=2)**2

  #               Fq = fid.Fidelity(rho)

  #               result_save['time'].append(time_all)
  #               result_save['epoch'].append(epoch)
  #               result_save['Fq'].append(Fq)
  #               result_save['loss'].append(loss-penalty)
  #               pbar.set_description(
  #                   "LBFGS_BM loss {:.10f} | Fq {:.8f} | time {:.5f}".format(loss-penalty, Fq, time_all))
  #               for name, p in self.generator.named_parameters():
  #                 if p.grad is not None:
  #                   param_norm = p.grad.data.norm(2)
  #                   print('\n')
  #                   print(f'Epoch {epoch}, {name} grad norm: {param_norm}')
  #           if Fq >= 0.9999:
  #               break

  #   pbar.close()
