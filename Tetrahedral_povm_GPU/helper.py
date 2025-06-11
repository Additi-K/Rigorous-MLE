import pickle
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

def adjust_after_restart(values):
    adjusted_values = []
    last_valid = 0  # Store the last value before the restart

    for i, val in enumerate(values):
        if i > 0 and val < values[i - 1]:  # Detect restart
            last_valid = values[i - 1]  # Store last value before restart

        if last_valid:
            adjusted_values.append(val + last_valid)
        else:
            adjusted_values.append(val)

    return adjusted_values

def plot_fval_vs_qubits(dir, save_destination=None):
    output = {}
    #get file names
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        # check for .pickle file
        if filename.endswith('.npy'):
            data = np.load(os.path.join(dir, filename), allow_pickle=True)
            data = data.item()
            args = (data['UGD']['parser'])
            n_qubits = args.n_qubits
            # print(n_qubits)
            n_samples = args.n_samples
            threshold = -1/n_samples * np.log(0.95)
            threshold = 1e-4
            # print(threshold)
            if n_qubits not in output:
                output[n_qubits] = {}
            fval_min = np.inf
            for algo in data.keys():
                fval = data[algo]['loss']
                if len(fval) != 0:
                  fval_min = torch.minimum(torch.tensor(fval_min), torch.min(torch.tensor(fval)))
                  # print(torch.min(torch.tensor(fval)), algo)

            for algo in data.keys():
              if algo == 'APG':
                data[algo]['time'] = adjust_after_restart(data[algo]['time'])
              fval = data[algo]['loss']
              condition = (torch.tensor(fval)-fval_min) <= threshold
              indices = torch.where(condition)[0]
              if len(indices) > 0:
                idx = indices[0].item()  # Get first occurrence of True
                # print(idx, algo)
                time_taken = data[algo]['time'][idx]
              else:
                # print(algo)
                time_taken = float('inf')

              output[n_qubits][algo] = time_taken


    res = {}
    algos = list(output[8].keys())
    for algo in algos:
      # print(algo.dtype)
      if algo == 'lbfgs':
        algo = 'L-BFGS'
      elif algo == 'APG':
        algo = 'CG-APG'

      res.setdefault(algo, {})

    # n_qubits = list(output.keys())

    for algo in algos:
      if algo == 'lbfgs':
        algo_ = 'L-BFGS'
      elif algo == 'APG':
        algo_ = 'CG-APG'
      else:
        algo_ = algo

      for key in output.keys():

        res[algo_][key] = output[key][algo]

    res.pop('LRE')
    res.pop('LRE_projA')

    # print(res)
    #Plot the data
    markers = {'L-BFGS':'v', 'CG-APG':'^', 'UGD': 'o', 'MGD': 'x', 'iMLE':'d', 'LRE': 's', 'LRE_projA':'p'}
    colors = {'L-BFGS':'#8c564b', 'CG-APG':'#9467bd', 'UGD': '#1f77b4', 'MGD': '#ff7f0e', 'iMLE':'#2ca02c', 'LRE': '#d62728', 'LRE_projA':'#e377c2'}
    fontSize=12
    plt.rcParams.update({
    'font.size': fontSize,         # Set font size for labels, legends, and ticks
    'axes.labelsize': fontSize,    # X and Y labels
    'legend.fontsize': fontSize,   # Legend
    'xtick.labelsize': fontSize,   # X-axis tick labels
    'ytick.labelsize': fontSize    # Y-axis tick labels
    })
    # plt.figure(figsize=(10, 6))

    for algo, results in res.items():
      if algo != 'LRE' or 'LRE_projA':
          # Sort inner dictionary by key
          sorted_items = sorted(results.items())

          x, y = zip(*[(k, v) for k, v in sorted_items if k != 12])  # Filter out inf values

          # if x:  # Only plot if there are valid values
          ax = plt.semilogy(x, y, color = colors[algo] ,  marker=markers[algo], markerfacecolor='none', label=algo)



    plt.xlabel('Number of Qubits')
    plt.ylabel('Time (s)')
    plt.xticks([8, 9, 10 ,11])
    plt.grid(True, which='both', linestyle='--', alpha = 1.0 , linewidth = 0.3, dashes=(2, 10))
    plt.title("Algorithm Performance Comparison")
    plt.legend(loc = 'upper left')


    # if save_destination != None:
    plt.savefig('fval_vs_qubit_hist_10_low_acc_pure_100x.pdf', format='pdf')
    plt.show()




def plot_error_vs_iteration(dir, n, save_destination=None):
    output = {}
    #get file names
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        # check for .pickle file
        if filename.endswith('.npy'):
            data = np.load(os.path.join(dir, filename), allow_pickle=True)
            data = data.item()
            args = (data['UGD']['parser'])
            n_qubits = args.n_qubits
            if n_qubits == n:
              #  data_ = data
              #  args_ = args
              break
    # print(n_qubits)
    n_samples = args.n_samples
    threshold_upp = -1/n_samples * np.log(0.95)
    threshold_low = 1e-4
    # print(threshold)
    if n_qubits not in output:
        output[n_qubits] = {}
    fval_min = np.inf
    for algo in data.keys():
        fval = data[algo]['loss']
        if len(fval) != 0:
          fval_min = torch.minimum(torch.tensor(fval_min), torch.min(torch.tensor(fval)))
          # print(torch.min(torch.tensor(fval)), algo)

    for algo in data.keys():
      # if algo == 'APG':
      #   data[algo]['time'] = adjust_after_restart(data[algo]['time'])
      fval = data[algo]['loss']
      error = torch.tensor(fval) - fval_min

      output[n_qubits][algo] = np.abs(error)


    res = {}
    algos = list(output[n].keys())
    for algo in algos:
      # print(algo.dtype)
      if algo == 'lbfgs':
        algo = 'L-BFGS'
      elif algo == 'APG':
        algo = 'CG-APG'

      res.setdefault(algo, {})


    for algo in algos:
      if algo == 'lbfgs':
        algo_ = 'L-BFGS'
      elif algo == 'APG':
        algo_ = 'CG-APG'
      else:
        algo_ = algo

      for key in output.keys():

        res[algo_][key] = output[key][algo]

    res.pop('LRE')
    res.pop('LRE_projA')

    #Plot the data
    markers = {'L-BFGS':'v', 'CG-APG':'^', 'UGD': 'o', 'MGD': 'x', 'iMLE':'d', 'LRE': 's', 'LRE_projA':'p'}
    colors = {'L-BFGS':'#8c564b', 'CG-APG':'#9467bd', 'UGD': '#1f77b4', 'MGD': '#ff7f0e', 'iMLE':'#2ca02c', 'LRE': '#d62728', 'LRE_projA':'#e377c2'}
    fontSize=12
    plt.rcParams.update({
    'font.size': fontSize,         # Set font size for labels, legends, and ticks
    'axes.labelsize': fontSize,    # X and Y labels
    'legend.fontsize': fontSize,   # Legend
    'xtick.labelsize': fontSize,   # X-axis tick labels
    'ytick.labelsize': fontSize    # Y-axis tick labels
    })
    # plt.figure(figsize=(10, 6))

    for algo, results in res.items():
      if algo != 'LRE' or 'LRE_projA':
          plt.plot(np.arange(results[n].shape[0]), results[n],  label=algo, color=colors[algo])  
    plt.axhline(y=threshold_low, color='black', linestyle='--', linewidth=1, label='Low-acc margin')
    plt.axhline(y=threshold_upp, color='gray', linestyle='--', linewidth=1, label='High-acc margin' )        
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend(ncol=2)
    plt.grid(True, which='both', linestyle='--', alpha = 1.0 , linewidth = 0.3, dashes=(2, 10))
    # plt.legend(loc = 'upper left')


    if save_destination != None:
      plt.savefig(save_destination + f'/error_vs_iteration_hist_10_n_{n}_dep_0.0.pdf', format='pdf')
      plt.show()
    
    
    
    
