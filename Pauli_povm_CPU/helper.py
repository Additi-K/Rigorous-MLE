
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import cvxpy as cp
import utils
from utils import *
import json
import os
import pickle
import time



def run_experiment(algo_list, save_dir = None, file_type = None, **param):
    results = {}
    for i, algo in enumerate(algo_list):
        results[algo.__name__] = algo(**param)

    results = {**results, **{'nQubits':param['nQubits'],
                    'warmstart':param['warmstart'],
                    'r_svd':param['r_svd'],
                    'eta':param['eta'],
                    'eps_coeff': param['eps_coeff'],
                    'c_eps': param['c_eps'],
                    'n_rate':param['n_rate'],
                    'rho_true': param['rho_true'],
                    'y' : param['y'],
                    'time_limit': param['timeLimit'] }}
    if param['save']:
        save_experiment(results, save_dir, file_type, **param)
    
    return results

def save_experiment(file, dir, format, **param):
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    if param['state'] == 'Wstate':
        file_path = os.path.join(dir, 'n_'+ str(param['nQubits']) + '_'
                                  + str(param['state']) + '_'
                                  + 'depolarize_' + str(param['depolarize']))

    else:
        file_path = os.path.join(dir, 'n_'+ str(param['nQubits']) + '_'
                                  + 'depolarize_' + str(param['depolarize']))

    start_time = time.time()
    if format == 'json':
        with open(file_path + '.json', 'w') as f:
            json.dump(file, f, indent=4)
    elif format == 'pickle':
        with open(file_path + '.pickle', 'wb') as f:
            pickle.dump(file, f)
    print('time for saving output:', time.time()-start_time)


def plot_fval_vs_qubits(dir, algo_list, save_destination=None):
    output = {algo.__name__: [] for algo in algo_list}
    nQubits = []
    #get file names
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        # check for .pickle file
        if filename.endswith('.pickle'):
            #read content
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                nQubits.append(data['nQubits'])
                # print(data['nQubits'])
                fval_min = np.inf
                for algo in output.keys():
                    if algo in data:
                        fval = data[algo]['fval']
                        fval_min = np.minimum(fval_min, np.min(fval))
                        # print(algo, np.min(fval))
                # print(fval_min)       
                N = 4**(nQubits[-1])* 100
                # print(-1/N * np.log(0.95))
                # print('qubits=', {nQubits[-1]})        
                for algo in output.keys():
                    
                    if algo in data: 
                        # if algo == 'LBFGS':
                        #     fval = data[algo]['fval'] - np.sum(data['y'])
                        fval = data[algo]['fval']
                        # print(f'{algo=}, {np.min(fval)=}')
                        condition = (fval-fval_min) <= 1e-4 #-1/N * np.log(0.95) # 1e-4
                        idx = np.argmax(condition)
                        cumulative_time = data[algo]['elapsed_time']
                        time_taken = cumulative_time[idx] if condition[idx] else np.inf
                        output[algo].append(time_taken)
                    else:
                        output[algo].append(np.inf)


    sorted_indices = np.argsort(nQubits)
    sorted_n = np.array(nQubits)[sorted_indices]

    # Reorder each algorithm's data
    sorted_output_dict = {key: np.array(values)[sorted_indices] for key, values in output.items()}
    # print(sorted_output_dict)

    # Plot the data
    markers = ['o', 'd', 's', 'x', '^', 'v']
    # plt.figure(figsize=(10, 6))
    fontSize=12
    plt.rcParams.update({
    'font.size': fontSize,         # Set font size for labels, legends, and ticks
    'axes.labelsize': fontSize,    # X and Y labels
    'legend.fontsize': fontSize,   # Legend
    'xtick.labelsize': fontSize,   # X-axis tick labels
    'ytick.labelsize': fontSize    # Y-axis tick labels
    })
    for i, (algo, values) in enumerate(sorted_output_dict.items()):
        plt.semilogy(sorted_n, values, marker=markers[i % len(markers)], markerfacecolor='none')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Time (s)')
    plt.legend(["MD", "low-rank MEG", "LBSDA", "d-sample LBSDA", "CG-APG", "L-BFGS", 'Acc-GD'], loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha = 1.0 , linewidth = 0.3, dashes=(2, 10))
    if save_destination != None:
        plt.savefig(save_destination+'/fval_vs_qubit_maxcor_2_high_acc_new.pdf', format='pdf')
    plt.show()

def plot_error_vs_iter(dir, algo_list, n, save_destination=None):
    output = {algo.__name__: [] for algo in algo_list}
    nQubits = []
    # Plot the data
    markers = ['o', 'd', 's', 'x', '^', 'v']
    marker_iter = iter(markers)
    # plt.figure(figsize=(10, 6))
    fontSize=10
    plt.rcParams.update({
    'font.size': fontSize,         # Set font size for labels, legends, and ticks
    'axes.labelsize': fontSize,    # X and Y labels
    'legend.fontsize': fontSize,   # Legend
    'xtick.labelsize': fontSize,   # X-axis tick labels
    'ytick.labelsize': fontSize    # Y-axis tick labels
    })
    map = {'EMD': 'MD', 'approx_MEG': 'low-rank MEG', 'LBSDA':'LBSDA', 'd_LBSDA':'d-sample LBSDA', 'qse_apg':'CG-APG', 'LBFGS':'L-BFGS', 'AccGD':'Acc-GD'}

    #get file names
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        # check for .pickle file
        if filename.endswith('.pickle'):
            #read content
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if data['nQubits'] == n:
                    data_n = data
                    print(data_n.keys())
                    
    fval_min = np.inf
    for algo in output.keys():
        if algo in data_n:
            fval = data_n[algo]['fval']
            fval_min = np.minimum(fval_min, np.min(fval))      
    
    N = 4**(n)* 100
    
    for algo in output.keys():
        if algo in data_n: 
            fval = data_n[algo]['fval']
            error = fval-fval_min #-1/N * np.log(0.95), 1e-4
            output[algo] = np.abs(error)

    
    plt.figure()
    for algo_name, err in output.items():
        plt.plot(np.arange(len(err)), err,  label=map[algo_name])  
    plt.axhline(y=1e-4, color='black', linestyle='--', linewidth=1, label='Low-acc margin')
    plt.axhline(y=-1/N * np.log(0.95), color='gray', linestyle='--', linewidth=1, label='High-acc margin' )        
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend(ncol=2)
    print(save_destination+'/error_vs_iter_maxcor_2.pdf')
    # plt.legend(["MD", "low-rank MEG", "LBSDA", "d-sample LBSDA", "CG-APG", "L-BFGS", 'Acc-GD'], loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha = 1.0 , linewidth = 0.3, dashes=(2, 10))
    if save_destination != None:
        plt.savefig(save_destination+f'/error_vs_iter_maxcor_2_n_{n}.pdf', format='pdf')
    plt.show()




'''
def plot_fidelity(dir, algo_list):

    output = {algo.__name__: [] for algo in algo_list}
    color_map = {'EMD': 'green', 'approx_MEG':'blue', 'LBSDA':'orange',
                  'd_LBSDA':'saddlebrown', 'qse_apg':'magenta', 'LBFGS':'black'}
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        # check for .pickle file
        if filename.endswith('.pickle'):
            #read content
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            plt.figure()
            for algo in output.keys():
                if algo in data:   
                    plt.plot(data[algo]['elapsed_time'], data[algo]['fidelity'], label= algo, color =color_map[algo], linewidth=1)  
            plt.legend()
            plt.title(str(data['nQubits'])) 


def plot_fval(dir, algo_list):

    output = {algo.__name__: [] for algo in algo_list}
    color_map = {'EMD': 'green', 'approx_MEG':'blue', 'LBSDA':'orange',
                  'd_LBSDA':'saddlebrown', 'qse_apg':'magenta', 'LBFGS':'black'}
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        # check for .pickle file
        if filename.endswith('.pickle'):
            #read content
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            fval_min = np.inf
            for algo in output.keys():
                if algo in data:
                    fval = data[algo]['fval']
                    fval_min = np.minimum(fval_min, np.min(fval))    
            nQubits = data['nQubits']   
            nShots = 100*(4**nQubits) 
            plt.figure()
            for algo in output.keys():
                if algo in data:   
                    # print(-nShots*data[algo]['fval'])
                    # plt.plot(np.array(data[algo]['elapsed_time']), np.exp(np.array(-nShots*(data[algo]['fval']-fval_min))), label= algo, color =color_map[algo], linewidth=1)
                    plt.semilogy(data[algo]['elapsed_time'], data[algo]['fval']-fval_min, label= algo, color =color_map[algo], linewidth=1)  
            plt.legend()
            plt.title('fval, n = ' + str(data['nQubits']))                 
'''

            
        