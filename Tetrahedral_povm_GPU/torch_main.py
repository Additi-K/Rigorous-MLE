# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-05-20 18:58:08
# @Last Modified by:   yong
# @Last Modified time: 2024-07-30 15:41:06

"""
-----------------------------------------------------------------------------------------
    The main function of quantum state tomography, used in the experimental 
    part of the paper, calls other implementations of the QST algorithm,
    paper: ``# @Paper: Efficient factored gradient descent algorithm for quantum state tomography``.

    @article{PhysRevResearch.6.033034,
      title = {Efficient factored gradient descent algorithm for quantum state tomography},
      author = {Wang, Yong and Liu, Lijun and Cheng, Shuming and Li, Li and Chen, Jie},
      journal = {Phys. Rev. Res.},
      volume = {6},
      issue = {3},
      pages = {033034},
      numpages = {11},
      year = {2024},
      month = {Jul},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevResearch.6.033034},
      url = {https://link.aps.org/doi/10.1103/PhysRevResearch.6.033034}
    }
-----------------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import torch
import numpy as np

from Basis.Basis_State import Mea_basis, State
from Basis.Basic_Function import get_default_device
from evaluation.Fidelity import Fid
from datasets.dataset import Dataset_P, Dataset_sample, Dataset_sample_P
from models.others.LRE import LRE
from models.others.qse_apg import qse_apg
from models.others.iMLE import iMLE
from models.UGD.ugd import UGD_nn, UGD
from models.others.lbfgs_bm import lbfgs_nn, lbfgs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def Net_train(opt, device, r_path, rho_star=None):
    """
    *******Main Execution Function*******
    """
    torch.cuda.empty_cache()
    print('\nparameter:', opt)

    # ----------file----------
    if os.path.isdir(r_path):
        print('result dir exists, is: ' + r_path)
    else:
        os.makedirs(r_path)
        print('result dir not exists, has been created, is: ' + r_path)

    # ----------rho_star and M----------
    print('\n'+'-'*20+'rho'+'-'*20)
    if rho_star is None:
        state_star, rho_star = State().Get_state_rho(
            opt.na_state, opt.n_qubits, opt.P_state, opt.rank)

    if opt.ty_state == 'pure':  # pure state
        rho_star = state_star

    rho_star = torch.from_numpy(rho_star).to(torch.complex64).to(device)
    M = Mea_basis(opt.POVM).M
    M = torch.from_numpy(M).to(device)

    # eigenvalues, _ = torch.linalg.eigh(rho_star)
    # print('-'*20, eigenvalues)

    # ----------data----------
    print('\n'+'-'*20+'data'+'-'*20)
    print('read original data')
    if opt.noise == "no_noise":  # perfect measurment
        print('----read ideal data')
        P_idxs, data, data_all = Dataset_P(
            rho_star, M, opt.n_qubits, opt.K, opt.ty_state, opt.P_povm, opt.seed_povm)
    else:
        print('----read sample data')
        if opt.P_povm == 1:  # measure all POVM
            P_idxs, data, data_all = Dataset_sample(opt.POVM, opt.na_state, opt.n_qubits,
                                                    opt.n_samples, opt.P_state, opt.ty_state,
                                                    rho_star, M, opt.read_data)
        else:  # measure partial POVM
            P_idxs, data, data_all = Dataset_sample_P(opt.POVM, opt.na_state, opt.n_qubits,
                                                      opt.K, opt.n_samples, opt.P_state,
                                                      opt.ty_state, rho_star, opt.read_data,
                                                      opt.P_povm, opt.seed_povm)

    in_size = len(data)
    print('data shape:', in_size)

    # fidelity
    fid = Fid(basis=opt.POVM, n_qubits=opt.n_qubits, ty_state=opt.ty_state,
              rho_star=rho_star, M=M, device=device)
    CF = fid.cFidelity_S_product(P_idxs, data)
    print('classical fidelity:', CF)

    # ----------------------------------------------QST algorithms----------------------------------------------------
    result_saves = {}
    
    
    # ---1: UGD with MRprop---
    print('\n'+'-'*20+'UGD_MRprop'+'-'*20)
    gen_net = UGD_nn(opt.na_state, opt.n_qubits, P_idxs, M,
                     map_method=opt.map_method, P_proj=opt.P_proj).to(torch.float32).to(device)

    net = UGD(opt.na_state, opt.map_method, gen_net, data, opt.lr, optim_f="M")
    result_save = {'parser': opt,
                   'time': [],
                   'epoch': [],
                   'Fq': [], 
                  'loss': []}
    net.train(opt.n_epochs, fid, result_save)
    result_saves['UGD'] = result_save

    # ---2: UGD with MGD---
    print('\n'+'-'*20+'MGD'+'-'*20)
    gen_net = UGD_nn(opt.na_state, opt.n_qubits, P_idxs, M,
                     map_method=opt.map_method, P_proj=opt.P_proj).to(torch.float32).to(device)

    net = UGD(opt.na_state, opt.map_method, gen_net, data, opt.lr, optim_f="S")
    result_save = {'parser': opt,
                   'time': [],
                   'epoch': [],
                   'Fq': [], 
                  'loss': []}
    net.train(opt.n_epochs, fid, result_save)
    result_saves['MGD'] = result_save

    # ---3: iMLE---
    print('\n'+'-'*20+'iMLE'+'-'*20)
    result_save = {'parser': opt,
                   'time': [],
                   'epoch': [],
                   'Fq': [], 
                  'loss': []}
    iMLE(M, opt.n_qubits, data_all, opt.n_epochs, fid, result_save, device)
    result_saves['iMLE'] = result_save

    # ---4: CG_APG---
    print('\n'+'-'*20+'CG-APG'+'-'*20)
    result_save = {'parser': opt,
                   'time': [],
                   'epoch': [],
                   'Fq': [], 
                  'loss': []}
    qse_apg(M, opt.n_qubits, data_all, opt.n_epochs,
            fid, 'proj_S', 2, result_save, device)
    result_saves['APG'] = result_save

    # ---5: LRE---
    print('\n'+'-'*20+'LRE'+'-'*20)
    result_save = {'parser': opt,
                   'time': [],
                   'Fq': [], 
                  'loss': []}
    LRE(M, opt.n_qubits, data_all, fid, 'proj_S', 1, result_save, device)
    result_saves['LRE'] = result_save

    # ---6: LRE with ProjA_1---
    print('\n'+'-'*20+'LRE proj'+'-'*20)
    result_save = {'parser': opt,
                   'time': [],
                   'Fq': [], 
                  'loss': []}
    LRE(M, opt.n_qubits, data_all, fid, 'proj_A', 1, result_save, device)
    result_saves['LRE_projA'] = result_save
    
    # ---7: LBFGS with BM---
    print('\n'+'-'*20+'lbfgs_bm'+'-'*20)
    gen_net = lbfgs_nn(opt.na_state, opt.n_qubits, P_idxs, M, opt.rank_lbfgs).to(torch.float32).to(device)

    net = lbfgs(opt.na_state, gen_net, data, opt.lr)
    result_save = {'parser': opt,
                   'time': [],
                   'epoch': [],
                   'Fq': [], 
                  'loss': []}
    net.train(opt.n_epochs, fid, result_save)
    result_saves['lbfgs'] = result_save

    return result_saves


if __name__ == '__main__':
    """
    *******Main Function*******
    Given QST perform parameters.
    """
    # ----------device----------
    print('-'*20+'init'+'-'*20)
    device = get_default_device()
    print('device:', device)

    # ----------parameters----------
    print('-'*20+'set parser'+'-'*20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--POVM", type=str, default="Tetra4", help="type of POVM")
    parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

    parser.add_argument("--n_qubits", type=int, default=6, help="number of qubits")
    parser.add_argument("--na_state", type=str, default="real_random_rank", help="name of state in library")
    parser.add_argument("--P_state", type=float, default=0.1, help="P of mixed state")
    parser.add_argument("--rank", type=float, default=2**5, help="rank of mixed state")
    parser.add_argument("--ty_state", type=str, default="mixed", help="type of state (pure, mixed)")

    parser.add_argument("--noise", type=str, default="noise", help="have or have not sample noise (noise, no_noise)")
    parser.add_argument("--n_samples", type=int, default=int(1e10), help="number of samples")
    parser.add_argument("--P_povm", type=float, default=1, help="possbility of sampling POVM operators")
    parser.add_argument("--seed_povm", type=float, default=1.0, help="seed of sampling POVM operators")
    parser.add_argument("--read_data", type=bool, default=False, help="read data from text in computer")

    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.1, help="optim: learning rate")
    parser.add_argument("--rank_lbfgs", type=int, default=1, help="rank for lbfgs reconstruction")

    parser.add_argument("--map_method", type=str, default="fac_h", 
                        help="map method for output vector to density matrix (fac_t, fac_h, fac_a, proj_M, proj_S, proj_A)")
    parser.add_argument("--P_proj", type=float, default=1, help="coefficient for proj method")
    parser.add_argument("--r_path", type=str, default="results/result/")

    opt = parser.parse_args()

    # r_path = 'results/result/' + opt.na_state + '/'
    # results = Net_train(opt, device, r_path)


    # -----ex: 0 (Convergence Experiment of W State for Different Qubits, noise, limited measurements, LBFGS included)-----

    r_path = opt.r_path + 'QST/data/tetra_4/'
    for n_qubit in [7, 8, 9, 10, 11]:
        opt.rank_lbfgs = 1
        opt.n_qubits = n_qubit
        opt.n_samples = 1 * (opt.K ** opt.n_qubits)
        save_data = {}
        results = Net_train(opt, device, r_path)

        np.save(r_path +  str(opt.na_state) + '_' + str(n_qubit) + '_' + str(opt.P_state) + '.npy', results)


    # -----ex: 1 (Tomography Convergence or Accuracy Experiments for Different Mapping Methods)-----
    '''
    opt.n_qubits = 8
    opt.ty_state = 'mixed'
    opt.na_state = 'real_random_pur_rank'
    opt.noise = 'noise'
    opt.n_epochs = 1000  # for MLE with UGD (MRprop), 5000 for MLE with UGD (MGD)
    dim = 2**opt.n_qubits
    r_path = 'results/result/' + opt.na_state + '/'

    for sample in [10**10]:
        opt.n_samples = sample
        
        for m_method in ['proj_A']:  # ['fac_a', 'fac_h', 'fac_t', 'proj_S', 'proj_M', 'proj_A']
            opt.map_method = m_method
            if m_method == 'proj_A':
                p_projs = [1]
            elif m_method in ('proj_S', 'proj_M'):
                p_projs = [1]
            else:
                p_projs = [1]

            for p_proj in p_projs:
                opt.P_proj = p_proj
                save_data = {}

                # Werner state with different p
                opt.na_state = 'real_random_pur'
                if m_method in ('proj_S', 'proj_M') and p_proj == 0:
                    opt.lr = 1e-3
                else:
                    opt.lr = 2  # 5 0.5
                prs = np.linspace(1 / dim, 0.99, 11)
                for idx, pr in enumerate(prs):
                    print('\n')
                    print('-'*20, idx + 1, pr)
                    P_s = np.sqrt((pr - 1 / dim) / (1 - 1 / dim))

                    e_f = 1
                    opt.P_state = P_s
                    while e_f:
                        try:
                            results = Net_train(opt, device, r_path)
                            e_f = 0
                        except:
                            e_f = 1

                    save_data[str(P_s)] = results

                # mixed state with different rank
                opt.na_state = 'real_random_rank'
                if m_method in ('proj_S', 'proj_M') and p_proj == 0:
                    opt.lr = 1e-3  # 1e-3
                else:
                    opt.lr = 5  # 1e-3 0.1
                logranks = np.linspace(0, 8, 9)
                for idx, logrank in enumerate(logranks):
                    print('\n')
                    print('-'*20, idx + 12, int(2**logrank))
                    P_s = int(2**logrank)

                    e_f = 1
                    opt.rank = P_s
                    while e_f:
                        try:
                            results = Net_train(opt, device, r_path)
                            e_f = 0
                        except:
                            e_f = 1

                    save_data[str(P_s)] = results

                np.save(r_path + 'UGD_' + m_method + '_' + str(p_proj) + '_' + str(int(np.log10(sample))) + '.npy', save_data)'''

    # -----ex: 2 (Eigenvalues analysis of different mapping methods)-----
    '''
    opt.n_qubits = 8
    opt.ty_state = 'mixed'
    opt.na_state = 'real_random_pur_rank'
    opt.noise = 'noise'
    opt.n_epochs = 1000
    opt.n_samples = 1e10
    dim = 2**opt.n_qubits
    r_path = 'results/result/' + opt.na_state + '/'

    opt.na_state = 'real_random_pur'
    prs = np.linspace(1 / dim, 0.99, 11)
    for idx, pr in enumerate(prs):
        P_s = np.sqrt((pr - 1 / dim) / (1 - 1 / dim))
        print('-'*20, idx, P_s, pr)
        opt.P_state = P_s
        save_data = {}
        _, rho_star = State().Get_state_rho(opt.na_state, opt.n_qubits, p=opt.P_state)
        save_data['actual'] = rho_star

        for m_method in ['fac_a', 'fac_h', 'fac_t', 'proj_S', 'proj_M', 'proj_A']:
            opt.map_method = m_method
            if m_method == 'proj_A':
                p_projs = [1, 4]
            elif m_method in ('proj_S', 'proj_M'):
                p_projs = [0, 1]
            else:
                p_projs = [1]

            for p_proj in p_projs:
                opt.P_proj = p_proj

                if m_method in ('proj_S', 'proj_M') and p_proj == 0:
                    opt.lr = 1e-3
                else:
                    opt.lr = 5  # 5

                results = Net_train(opt, device, r_path, rho_star)
                save_data[m_method+str(p_proj)] = results

        np.save(r_path + 'real_random_pur_rank' + "_" + str(round(pr, 4)) + '.npy', save_data)

    opt.na_state = 'real_random_rank'
    ranks = 2**np.linspace(0, 8, 9)
    for idx, P_s in enumerate([ranks[-1]]):
        print('-'*20, idx, P_s)
        opt.rank = int(P_s)
        save_data = {}
        _, rho_star = State().Get_state_rho(opt.na_state, opt.n_qubits, r=int(P_s))
        save_data['actual'] = rho_star

        for m_method in ['fac_a', 'fac_h', 'fac_t', 'proj_S', 'proj_M', 'proj_A']:
            opt.map_method = m_method
            if m_method == 'proj_A':
                p_projs = [1, 4]
            elif m_method in ('proj_S', 'proj_M'):
                p_projs = [0, 1]
            else:
                p_projs = [1]

            for p_proj in p_projs:
                opt.P_proj = p_proj

                if m_method in ('proj_S', 'proj_M') and p_proj == 0:
                    opt.lr = 1e-3
                else:
                    opt.lr = 1e-3  # 5

                results = Net_train(opt, device, r_path, rho_star)
                save_data[m_method+str(p_proj)] = results

        np.save(r_path + 'real_random_pur_rank' +
                "_" + str(int(P_s)) + '.npy', save_data)'''

    # -----ex: 3 (Random State Convergence Experiments of Different QST Algorithms for Different samples)-----
    '''
    opt.n_qubits = 8
    opt.na_state = 'real_random'
    opt.map_method = 'fac_h'
    opt.n_epochs = 10000
    r_path = 'results/result/' + opt.na_state + '/'
    for sample in [10**7, 10**8, 10**9, 10**10, 10**11]:
        if sample is not None:
            opt.n_samples = sample
            opt.noise = 'noise'
        else:
            opt.noise = 'no_noise'

        save_data = {}
        for idx, P_s in enumerate(np.random.uniform(0, 1, 20)):
            print('\n')
            print('-'*40, idx, P_s)

            e_f = 1
            while e_f:
                opt.P_state = P_s

                try:
                    results = Net_train(opt, device, r_path)
                    e_f = 0
                except:
                    P_s = np.random.uniform(0, 1, 1)[0]
                    e_f = 1

            save_data[str(P_s)] = results

        if sample is not None:
            np.save(r_path + str(int(np.log10(sample))) + 's_all.npy', save_data)
        else:
            np.save(r_path + '11' + 's_all.npy', save_data)'''

    # -----ex: 4 (Convergence Experiment of Random Mixed States for Different Qubits, no noise)-----
    '''
    opt.na_state = 'real_random'
    opt.noise = 'no_noise'
    opt.map_method = 'fac_h'
    opt.n_epochs = 10000
    r_path = 'results/result/' + opt.na_state + '/'
    for n_qubit in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        opt.n_qubits = n_qubit
        save_data = {}
        for idx, P_s in enumerate(np.random.uniform(0, 1, 20)):
            print('-'*40, idx, P_s)
            opt.P_state = P_s

            results = Net_train(opt, device, r_path)
            save_data[str(P_s)] = results

        np.save(r_path + str(n_qubit) + 'q_all' + '.npy', save_data)
    '''
