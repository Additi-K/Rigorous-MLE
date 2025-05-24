import numpy as np
import scipy
from scipy.sparse.linalg import svds
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from proxop import Simplex
from utils import *
from algos import *
from time import time
import sparse
import math
import sys



def proj_spectrahedron(Y):
    """
    Project a matrix onto the spectrahedron.
    
    Parameters:
    Y : ndarray
        Input Hermitian matrix.
    
    Returns:
    X : ndarray
        Positive semidefinite matrix X such that the trace of X is equal to 1 
        and the Frobenius norm between X and Hermitian matrix Y is minimized.
    """
    # Perform eigenvalue decomposition and remove the imaginary components
    # that arise from numerical precision errors
    eigvals, eigvecs = np.linalg.eig(Y)
    v = np.real(eigvals)
    # Project the eigenvalues onto the probability simplex
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.where(u > (sv - 1) / np.arange(1, len(u) + 1))[0][-1]
    theta_ = (sv[rho] - 1) / (rho+1)
    w = np.maximum(v - theta_, 0)

    # Element-wise multiplication with broadcasting
    X = eigvecs * np.sqrt(w[:, np.newaxis].T)

    # Matrix multiplication
    X = X @ X.T.conj()

    return X


def qse_apg(opts=None, **param):
    """
    QSE_APG Quantum state estimation via accelerated projected gradient (APG)
    """

    name = "CG_APG"
    print(name + " starts.")

    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y']
    nQubits = param['nQubits'] 
    timeLimit = param['timeLimit']


    d = 2**nQubits
    K = 2*(d**2 - 1) 
    # assert K == len(f)
    m = 4**nQubits - 1

    defaults = {
        'rho0': 'bootstrap',
        'threshold_step': 0.5 * np.sqrt(d) * d * np.finfo(float).eps,
        'threshold_fval': -np.sum(y[y != 0] * np.log(y[y != 0])),
        # 'threshold_fval': -np.sum(y[y != 0] * np.log((m)*y[y != 0])),
        # 'threshold_fval': -np.sum(y * np.log(y)),
        'threshold_dist': 0,
        'rho_star': None,
        'imax': 10000,
        'save_rhos': True,
        'guard': True,
        'bb': True,
        'accel': 'fista_tfocs',
        't0': None,
        'restart_grad_param': 0.01,
        'bfactor': 0.5,
        'afactor': 1.1,
        'minimum_t': np.finfo(float).eps,
        'bootstrap_threshold': 1e-2
    }
    print(defaults['threshold_fval'])
    if opts is None:

        opts = defaults
    else:
        for key, value in defaults.items():
            if key not in opts:
                opts[key] = value

    # fmap = y != 0
    # y= y[fmap]
    coeff_temp = np.zeros(K)

    stats = {
        'steps': np.zeros(opts['imax']),
        'fval': np.zeros(opts['imax']),
        'dists': np.zeros(opts['imax']),
        'dists_true': np.zeros(opts['imax'], dtype=bool),
        'comp_prob': 0,
        'comp_grad': 0,
        'comp_fval': 0,
        'comp_proj': 0,
        'elapsed_time': np.zeros(opts['imax']),
        'thetas': np.zeros(opts['imax']),
        'ts': np.zeros(opts['imax']),
        'satisfied_step': False,
        'satisfied_fval': False,
        'satisfied_dist': False,
        'best_rho': None,
        'best_fval': float('inf'),
        'rho': [] if opts['save_rhos'] else None,
        'fidelity': np.zeros(opts['imax']),
        'norm_err': np.zeros(opts['imax']),
        'n_epoch' : np.zeros(opts['imax'])
    }
    # start = time()
    
    if isinstance(opts['rho0'], str):
        if opts['rho0'] == 'white':
            rho = np.eye(d) / d
        elif opts['rho0'] == 'bootstrap':
            opts2 = {key: opts[key] for key in ['imax', 'threshold_fval', 'threshold_dist', 'rho_star', 'save_rhos']}
            opts2['mincondchange'] = opts['bootstrap_threshold']
            # coeff_temp[fmap] = y
            # coeff_temp = y
            # param['y'] = coeff_temp
            # time_offset = time() - start
            rho, boot_stats = qse_cgls(opts2, **param)
            # print(f'exit CG')
            opts['imax'] -= len(boot_stats['fval'])
            # print(opts['imax'])
            # boot_stats['elapsed_time'] += time_offset
            # boot_stats['elapsed_time'] = list(np.asarray(boot_stats['elapsed_time']) + 1)
            if boot_stats['satisfied_fval'] or boot_stats['satisfied_dist'] or boot_stats['elapsed_time'][-1]>=timeLimit:
                # print('only CG')
                stats.update(boot_stats)
                keys = ["n_epoch", "fidelity", "fval", "elapsed_time", 'rho', 'norm_err']
                return {k: stats[k] for k in keys if k in stats}
                # return rho, stats
        else:
            raise ValueError(f"Unknown initializer specification: {opts['rho0']}")
    else:
        rho = opts['rho0']

    varrho = rho.copy()
    varrho_changed = True
    gradient = None
    theta = 1
    t = opts['t0']
    fval = float('inf')
    bb_okay = False

    probs_rho = qst1(rho, nQubits)
    probs_rho = np.concatenate((0.5*(probs_rho[0] + probs_rho[1:]), 0.5*(probs_rho[0]-probs_rho[1:])))
    # probs_rho = probs_rho[fmap]
    stats['comp_prob'] += 1
    stats['comp_fval'] += 1
    probs_varrho = probs_rho
    name = "APG"
    print(name + " starts.")

    start = time()
    for i in range(opts['imax']):
        if i%10 == 0:
            print(i)
        # start = time.time()
        stats['n_epoch'][i] = i
        if varrho_changed:
            if opts['bb'] and i > 0:
                old_gradient = gradient

            # coeff_temp[fmap] = f / probs_varrho
            # coeff_temp[fmap] = y
            gradient = gradf(varrho, nQubits, y, primitive1 = qst1, primitive2 = qst2)
            # if not isinstance(gradient, np.ndarray):
            #     gradient = gradient.todense()
            stats['comp_grad'] += 1
            fval_varrho = fun(varrho, nQubits, y, primitive1 = qst1, primitive2 = qst2)
            stats['comp_fval'] += 1

            if opts['bb'] and bb_okay:
                # varrho_diff = varrho.ravel() - old_varrho.ravel()
                # gradient_diff = gradient.ravel() - old_gradient.ravel()
                varrho_diff = varrho - old_varrho
                gradient_diff = gradient - old_gradient
                denominator = np.vdot(gradient_diff, gradient_diff)
                if denominator > 0:
                    t = abs(np.real(np.vdot(varrho_diff, gradient_diff))) / denominator

        if i == 0 and 'boot_stats' in locals():
            boot_stats['fval'][-1] = np.real_if_close(fval_varrho)

        if t is None:
            # probs_gradient = qmt(-gradient, operators[fmap])
            probs_gradient = qst1(-gradient, nQubits)
            probs_gradient = np.concatenate((0.5*(probs_gradient[0] + probs_gradient[1:]), 0.5*(probs_gradient[0]-probs_gradient[1:])))
            # probs_gradient = probs_gradient[fmap]
            first_deriv = np.sum(-y * (probs_gradient / probs_varrho))
            second_deriv = np.sum(y * (probs_gradient ** 2 / probs_varrho ** 2))
            stats['comp_prob'] += 1
            t = -first_deriv / second_deriv
    
        else:
            if not (opts['bb'] and bb_okay):
                t *= opts['afactor']
              

        t_good = False
        fval_new = None
        t_threshold = opts['minimum_t'] / np.linalg.norm(gradient.ravel())
        while not t_good:
            if fval_new is not None:
                new_t_estimate = second_order / max(0, fval_new - fval_varrho - first_order)
                if new_t_estimate > t_threshold:
                    t = min(t * opts['bfactor'], new_t_estimate)
             
                else:
                    t *= opts['bfactor']
    

            rho_new = proj_spectrahedron(varrho - t * gradient)
            stats['comp_proj'] += 1
            # probs_rho_new = qmt(rho_new, operators[fmap])
            probs_rho_new = qst1(rho_new, nQubits)
            probs_rho_new = np.concatenate((0.5*(probs_rho_new[0] + probs_rho_new[1:]), 0.5*(probs_rho_new[0]-probs_rho_new[1:])))
            print(probs_rho_new)
            # probs_rho_new = probs_rho_new[fmap]
            fval_new = np.real_if_close(-np.sum(y * np.log(probs_rho_new)))
            # print(fval_new)
            stats['comp_prob'] += 1
            stats['comp_fval'] += 1
            stats['fval'][i] = fval_new
            stats['fidelity'][i] = fidelity(rho_new, rho_true)
            stats['norm_err'][i] = norm(rho_new-rho_true)
            # stats['fval'].append(fval_new)
            # delta = rho_new.ravel() - varrho.ravel()
            delta = rho_new- varrho
            first_order = np.real(np.vdot(gradient, delta))
            second_order = 0.5 * np.vdot(delta, delta)
            t_good = not (t * fval_new > t * fval_varrho + t * first_order + 0.9 * second_order)

        # if fval_new < stats['best_fval']:
        #     stats['best_fval'] = fval_new
        #     stats['best_rho'] = rho_new

        if opts['save_rhos']:
            stats['rho'] = rho_new

        stats['ts'][i] = np.real_if_close(t)
        stats['thetas'][i] = theta

        stats['steps'][i] = 0.5 * np.sqrt(d) * np.linalg.norm(rho_new - rho, 'fro')
        stats['satisfied_step'] = stats['steps'][i] <= opts['threshold_step']
        stats['satisfied_fval'] = fval_new <= opts['threshold_fval']

        if opts['rho_star'] is not None:
            stats['dists'][i] = 0.5 * np.linalg.norm(rho_new - opts['rho_star'], 'fro')
            if stats['dists'][i] <= opts['threshold_dist']:
                stats['dists'][i] = 0.5 * np.sum(np.linalg.svd(rho_new - opts['rho_star'])[1])
                stats['dists_true'][i] = True
                stats['satisfied_dist'] = stats['dists'][i] <= opts['threshold_dist']

        if t < t_threshold or stats['satisfied_step'] or stats['satisfied_fval'] or stats['satisfied_dist']:
            # print(t < t_threshold, stats['satisfied_step'], stats['satisfied_fval'], stats['satisfied_dist'])
            stats['elapsed_time'][i] = time() - start
            rho = rho_new
            # print('break')
            break

        if opts['bb']:
            old_varrho = varrho.copy()

        # vec1 = varrho.ravel() - rho_new.ravel()
        # vec2 = rho_new.ravel() - rho.ravel()
        vec1 = varrho - rho_new
        vec2 = rho_new - rho
        do_restart = np.real(np.vdot(vec1, vec2)) > -opts['restart_grad_param'] * np.linalg.norm(vec1) * np.linalg.norm(vec2)

        bb_okay = not do_restart

        if do_restart:
            varrho = rho.copy()
            probs_varrho = probs_rho.copy()
            varrho_changed = theta > 1
            theta = 1
            stats['elapsed_time'][i] = time() - start
            # stats['elapsed_time'].append(time.time() - start)
            stats['fval'][i] = fval
            stats['fidelity'][i] = fidelity(rho_new, rho_true)
            stats['norm_err'][i] = norm(rho_new- rho_true)
            if opts['save_rhos']:
                stats['rho'] = rho
            continue

        # Acceleration
        if i > 1 and stats['ts'][i] > np.finfo(float).eps:
            Lfactor = stats['ts'][i-1] / stats['ts'][i]
        else:
            Lfactor = 1

        if opts['accel'] == 'fista':
            theta_new = (1 + np.sqrt(1 + 4 * theta**2)) / 2
            beta = (theta - 1) / theta_new
            theta = theta_new
        elif opts['accel'] == 'fista_tfocs':
            theta_hat = np.sqrt(Lfactor) * theta
            theta_new = (1 + np.sqrt(1 + 4 * theta_hat**2)) / 2
            beta = (theta_hat - 1) / theta_new
            theta = theta_new
        elif opts['accel'] == 'none':
            beta = 0
        else:
            raise ValueError('Unknown acceleration scheme: {}'.format(opts['accel']))

        # Update
        varrho = rho_new + beta * (rho_new - rho)
        probs_varrho = probs_rho_new + beta * (probs_rho_new - probs_rho)
        if opts['guard'] and min(probs_varrho) <= 0:
            # Discard momentum if it causes varrho to become infeasible
            # Retain theta to keep estimate of current condition number
            varrho = rho_new
            probs_varrho = probs_rho_new

        varrho_changed = True
        rho = rho_new
        probs_rho = probs_rho_new
        fval = fval_new
        stats['elapsed_time'][i] = time() - start
        total_elapsed_time = boot_stats['elapsed_time'][-1] + stats['elapsed_time'][i]
        if total_elapsed_time >= timeLimit:
            print(f"Time limit exceeded: {total_elapsed_time:.2f}s > {timeLimit:.2f}s")
            break

    # Collect stats
    stats['n_epoch'] = stats['n_epoch'][:i+1]
    stats['fval'] = stats['fval'][:i+1]
    stats['fidelity'] = stats['fidelity'][:i+1]
    stats['norm_err'] = stats['norm_err'][:i+1]
    stats['steps'] = stats['steps'][:i+1]
    stats['dists'] = stats['dists'][:i+1]
    stats['dists_true'] = stats['dists_true'][:i+1]
    stats['elapsed_time'] = stats['elapsed_time'][:i+1]
    stats['thetas'] = stats['thetas'][:i+1]
    stats['ts'] = stats['ts'][:i+1]

    if 'boot_stats' in locals():
        # if opts['save_rhos']:
        #     stats['rhos'] = np.concatenate((boot_stats['rhos'], stats['rhos']))
        stats['fval'] = np.concatenate((boot_stats['fval'], stats['fval']))
        stats['elapsed_time'] = np.concatenate((boot_stats['elapsed_time'], stats['elapsed_time']))
        stats['fidelity'] = np.concatenate((boot_stats['fidelity'],  stats['fidelity']))
        stats['norm_err'] = np.concatenate((boot_stats['norm_err'],  stats['norm_err']))
        stats['n_epoch'] = np.concatenate((boot_stats['n_epoch'],  boot_stats['n_epoch'][-1] + 1+ stats['n_epoch']))
    

        # stats['steps'] = np.concatenate((boot_stats['steps'], stats['steps']))
        # stats['dists'] = np.concatenate((boot_stats['dists'], stats['dists']))
        # stats['dists_true'] = np.concatenate((boot_stats['dists_true'], stats['dists_true']))
        # stats['times'] = np.concatenate((boot_stats['times'], stats['times']))
        # stats['comp_prob'] += boot_stats['comp_prob']
        # stats['comp_grad'] += boot_stats['comp_grad']
        # stats['comp_fval'] += boot_stats['comp_fval']
        # stats['comp_proj'] += boot_stats['comp_proj']

        keys = ["n_epoch", "fidelity", "fval", "elapsed_time", 'rho', 'norm_err']
    return {k: stats[k] for k in keys if k in stats}


def qse_cgls(opts=None, **param):
    """
    QSE_CGLS Quantum state estimation via conjugate gradients with line search
    """

    name = "CG"
    print(name + " starts.")

    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y']
    nQubits = param['nQubits']
    timeLimit = param['timeLimit']

    d = 2**nQubits
    K = 2*(d**2 - 1)    
    m = 4**nQubits - 1 
    # assert K == len(f)

    defaults = {
        'threshold_step': 0.5 * np.sqrt(d) * d * np.finfo(float).eps,
        'threshold_fval': -np.sum(y[y != 0] * np.log(y[y != 0])),
        # 'threshold_fval': -np.sum(y[y != 0] * np.log((m)*y[y != 0])),
        # 'threshold_fval': -np.sum(y * np.log(y)),
        'threshold_dist': 0,
        'rho_star': None,
        'imax': 20000,
        'rho0': None,
        'save_rhos': False,
        'adjustment': 0.5,
        'mincondchange': -np.inf,
        'step_adjust': 2,
        'a2': 0.1,
        'a3': 0.2
    }

    if opts is None:
        opts = defaults
    else:
        for key, value in defaults.items():
            if key not in opts:
                opts[key] = value
    if opts['rho0'] is not None:
        V, D = np.linalg.eigh(opts['rho0'])
        temp = np.dot(np.diag(np.sqrt(np.maximum(0, np.diag(D)))), V.T)
        A = temp / np.linalg.norm(temp, 'fro')
        rho = A.T.conj()@ A
    else:
        A = np.eye(d) / np.sqrt(d)
        rho = np.eye(d) / d

    a2 = opts['a2']
    a3 = opts['a3']

    stats = {
        'steps': np.zeros(opts['imax']),
        'fval': np.zeros(opts['imax']),
        'dists': np.zeros(opts['imax']),
        'dists_true': np.zeros(opts['imax'], dtype=bool),
        'comp_prob': 0,
        'comp_grad': 0,
        'comp_fval': 0,
        'comp_proj': 0,
        'elapsed_time': np.zeros(opts['imax']),
        'satisfied_step': False,
        'satisfied_fval': False,
        'satisfied_dist': False,
        'best_rho': None,
        'best_fval': np.inf,
        'fidelity': np.zeros(opts['imax']), 
        'norm_err' : np.zeros(opts['imax']),
        'n_epoch': np.zeros(opts['imax'])
    }

    if opts['save_rhos']:
        stats['rhos'] = []

    # start = time.time()

    probs = qst1(rho, nQubits)
    probs = np.concatenate((0.5*(probs[0] + probs[1:]), 0.5*(probs[0]-probs[1:])))
    stats['comp_prob'] += 1
    # adj = y / probs
    # # adj = y
    # adj[y == 0] = 0
    # # rmatrix = qmt(adj, operators, 'adjoint')
    rmatrix = -gradf(rho, nQubits, y, primitive1 = qst1, primitive2 = qst2)

    condchange = np.inf

    if opts['mincondchange'] > 0:
        hessian_proxy = y / (probs) ** 2
        # hessian_proxy[y == 0] = 0

    stats['comp_grad'] += 1
    # fval = -np.sum(y[y != 0] * np.log(probs[y != 0]))
    fval = -np.sum(y * np.log(probs))
    # print(np.sum(probs))
    # fval = fun(rho, nQubits, f, operators)

    stats['best_rho'] = rho
    stats['best_fval'] = fval

    i=0
    start = time()
    for i in range(opts['imax']):
        print(i)
        stats['n_epoch'][i] = i
        curvature_too_large = False

        if opts['mincondchange'] > 0:
            if i > 0:
                condchange = np.real(
                    np.arccos(np.real(np.vdot(old_hessian_proxy, hessian_proxy)) /
                              np.linalg.norm(old_hessian_proxy.ravel()) / np.linalg.norm(hessian_proxy.ravel())))
            # stats['cond_change_angle'][i, 1] = condchange

        if i == 0:
            # G = np.dot(A, rmatrix - np.eye(d))
            G = A@(rmatrix - np.eye(d))
            H = G
        else:
            G_next = A@(rmatrix - np.eye(d))
            polakribiere = np.real(
                np.vdot(G_next, (G_next- opts['adjustment'] * G))
            ) / np.vdot(G, G)
            gamma = max(polakribiere, 0)
            H = G_next + gamma * H
            G = G_next

        A2 = A + a2 * H
        A3 = A + a3 * H
        rho2 = A2.T.conj()@ A2
        rho2 = rho2 / np.trace(rho2)
        rho3 = A3.T.conj()@A3
        rho3 = rho3 / np.trace(rho3)
        # probs2 = qst1(rho2, nQubits)
        # probs2 = np.concatenate((0.5*(probs2[0] + probs2[1:]), (0.5*probs2[0]-probs2[1:])))
        # probs3 = qst1(rho3, nQubits)
        # probs3 = np.concatenate((0.5*(probs3[0] + probs3[1:]), (0.5*probs3[0]-probs3[1:])))
        stats['comp_prob'] += 2
        l1 = fval
        # l2 = -np.sum(y[y != 0] * np.log(probs2[y != 0]))
        # l3 = -np.sum(y[y != 0] * np.log(probs3[y != 0]))
        l2 = fun(rho2, nQubits, y, primitive1 = qst1, primitive2 = qst2)
        l3 = fun(rho3, nQubits, y, primitive1 = qst1, primitive2 = qst2)
        stats['comp_fval'] += 2
        alphaprod = 0.5 * ((l3 - l1) * a2 ** 2 - (l2 - l1) * a3 ** 2) / ((l3 - l1) * a2 - (l2 - l1) * a3)
        if np.isnan(alphaprod) or alphaprod > 1 / np.finfo(float).eps or alphaprod < 0:
            candidates = [0, a2, a3]
            index = np.argmin([l1, l2, l3])
            if opts['step_adjust'] > 1:
                if np.isnan(alphaprod) or alphaprod > 1 / np.finfo(float).eps:
                    a2 *= opts['step_adjust']
                    a3 *= opts['step_adjust']
                elif alphaprod < 0:
                    a2 /= opts['step_adjust']
                    a3 /= opts['step_adjust']
                    curvature_too_large = True
            alphaprod = candidates[index]

        A = A + alphaprod * H
        A = A / np.linalg.norm(A, 'fro')
        old_rho = rho
        rho = A.T.conj()@ A

        if opts['save_rhos']:
            stats['rhos'].append(rho)

        # stats['alphas'][i] = alphaprod

        # probs = qst1(rho, nQubits)
        stats['comp_prob'] += 1
        fval = fun(rho, nQubits, y, primitive1 = qst1, primitive2 = qst2)
        stats['elapsed_time'][i] = time() - start
        stats['comp_fval'] += 1
        stats['fval'][i] = np.real_if_close(fval)
        stats['norm_err'][i] = norm(rho-rho_true)
        stats['fidelity'][i] = fidelity(rho, rho_true)

        # if stats['fval'][i] < stats['best_fval']:
        #     stats['best_fval'] = stats['fval'][i]
        #     stats['best_rho'] = rho

        if stats['elapsed_time'][i] >= timeLimit:
            print(f"Time limit exceeded: {stats['elapsed_time'][i]:.2f}s > {timeLimit:.2f}s")
            break
        srt = time()
        stats['steps'][i] = 0.5 * np.sqrt(d) * np.linalg.norm(rho - old_rho, 'fro')
        stats['satisfied_step'] = stats['steps'][i] <= opts['threshold_step']
        
        stats['satisfied_fval'] = fval <= opts['threshold_fval']
        if opts['rho_star'] is not None:
            stats['dists'][i] = 0.5 * np.linalg.norm(rho - opts['rho_star'], 'fro')
            if stats['dists'][i] <= opts['threshold_dist']:
                stats['dists'][i] = 0.5 * np.sum(np.linalg.svd(rho - opts['rho_star'])[1])
                stats['dists_true'][i] = True
                stats['satisfied_dist'] = stats['dists'][i] <= opts['threshold_dist']

        if (not curvature_too_large and stats['satisfied_step']) or stats['satisfied_fval'] or condchange < opts['mincondchange'] or stats['satisfied_dist']:
            break
        
        if i != opts['imax'] - 1:
            # adj = y / probs
            # # adj = y
            # adj[y == 0] = 0
            rmatrix = -grad(rho, nQubits, y, primitive1 = qst1, primitive2 = qst2)
            stats['comp_grad'] += 1
            if opts['mincondchange'] > 0:
                old_hessian_proxy = hessian_proxy
                hessian_proxy = y / (probs) ** 2
                # hessian_proxy[y == 0] = 0

    stats['n_epoch'] = stats['n_epoch'][:i+1]
    stats['fval'] = stats['fval'][:i+1]
    stats['fidelity'] = stats['fidelity'][:i+1]
    stats['norm_err'] = stats['norm_err'][:i+1]
    stats['steps'] = stats['steps'][:i+1]
    stats['dists'] = stats['dists'][:i+1]
    stats['dists_true'] = stats['dists_true'][:i+1]
    stats['elapsed_time'] = stats['elapsed_time'][:i+1]
    return rho, stats
'''


def qse_apg(opts=None, **param):
    """
    QSE_APG Quantum state estimation via accelerated projected gradient (APG)
    """

    name = "CG_APG"
    print(name + " starts.")

    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y']
    nQubits = param['nQubits'] 
    timeLimit = param['timeLimit']


    d = 2**nQubits
    K = 2*(d**2 - 1) 
    # assert K == len(f)
    m = 4**nQubits - 1

    defaults = {
        'rho0': 'bootstrap',
        'threshold_step': 0.5 * np.sqrt(d) * d * np.finfo(float).eps,
        'threshold_fval': -np.sum(y[y != 0] * np.log(y[y != 0])),
        # 'threshold_fval': -np.sum(y[y != 0] * np.log((m)*y[y != 0])),
        # 'threshold_fval': -np.sum(y * np.log(y)),
        'threshold_dist': 0,
        'rho_star': None,
        'imax': 10000,
        'save_rhos': True,
        'guard': True,
        'bb': True,
        'accel': 'fista_tfocs',
        't0': None,
        'restart_grad_param': 0.01,
        'bfactor': 0.5,
        'afactor': 1.1,
        'minimum_t': np.finfo(float).eps,
        'bootstrap_threshold': 1e-2
    }
    print(defaults['threshold_fval'])
    if opts is None:

        opts = defaults
    else:
        for key, value in defaults.items():
            if key not in opts:
                opts[key] = value

    # fmap = y != 0
    # y= y[fmap]
    coeff_temp = np.zeros(K)

    stats = {
        'steps': np.zeros(opts['imax']),
        'fval': np.zeros(opts['imax']),
        'dists': np.zeros(opts['imax']),
        'dists_true': np.zeros(opts['imax'], dtype=bool),
        'comp_prob': 0,
        'comp_grad': 0,
        'comp_fval': 0,
        'comp_proj': 0,
        'elapsed_time': np.zeros(opts['imax']),
        'thetas': np.zeros(opts['imax']),
        'ts': np.zeros(opts['imax']),
        'satisfied_step': False,
        'satisfied_fval': False,
        'satisfied_dist': False,
        'best_rho': None,
        'best_fval': float('inf'),
        'rho': [] if opts['save_rhos'] else None,
        'fidelity': np.zeros(opts['imax']),
        'norm_err': np.zeros(opts['imax']),
        'n_epoch' : np.zeros(opts['imax'])
    }
    start = time()
    
    if isinstance(opts['rho0'], str):
        if opts['rho0'] == 'white':
            rho = np.eye(d) / d
        elif opts['rho0'] == 'bootstrap':
            opts2 = {key: opts[key] for key in ['imax', 'threshold_fval', 'threshold_dist', 'rho_star', 'save_rhos']}
            opts2['mincondchange'] = opts['bootstrap_threshold']
            # coeff_temp[fmap] = y
            # coeff_temp = y
            # param['y'] = coeff_temp
            time_offset = time() - start
            rho, boot_stats = qse_cgls(opts2, **param)
            # print(f'exit CG')
            opts['imax'] -= len(boot_stats['fval'])
            # print(opts['imax'])
            boot_stats['elapsed_time'] += time_offset
            # boot_stats['elapsed_time'] = list(np.asarray(boot_stats['elapsed_time']) + 1)
            if boot_stats['satisfied_fval'] or boot_stats['satisfied_dist'] or (boot_stats['elapsed_time'][-1] - time_offset)>=timeLimit:
                # print('only CG')
                stats.update(boot_stats)
                keys = ["n_epoch", "fidelity", "fval", "elapsed_time", 'rho', 'norm_err']
                return {k: stats[k] for k in keys if k in stats}
                # return rho, stats
        else:
            raise ValueError(f"Unknown initializer specification: {opts['rho0']}")
    else:
        rho = opts['rho0']

    varrho = rho.copy()
    varrho_changed = True
    gradient = None
    theta = 1
    t = opts['t0']
    fval = float('inf')
    bb_okay = False

    probs_rho = qst1(rho, nQubits)
    probs_rho = np.concatenate((0.5*(probs_rho[0] + probs_rho[1:]), 0.5*(probs_rho[0]-probs_rho[1:])))
    # probs_rho = probs_rho[fmap]
    stats['comp_prob'] += 1
    stats['comp_fval'] += 1
    probs_varrho = probs_rho
    name = "APG"
    print(name + " starts.")

    # start = time()
    for i in range(opts['imax']):
        if i%10 == 0:
            print(i)
        # start = time.time()
        stats['n_epoch'][i] = i
        if varrho_changed:
            if opts['bb'] and i > 0:
                old_gradient = gradient

            # coeff_temp[fmap] = f / probs_varrho
            # coeff_temp[fmap] = y
            gradient = gradf(varrho, nQubits, y, primitive1 = qst1, primitive2 = qst2)
            # if not isinstance(gradient, np.ndarray):
            #     gradient = gradient.todense()
            stats['comp_grad'] += 1
            fval_varrho = fun(varrho, nQubits, y, primitive1 = qst1, primitive2 = qst2)
            stats['comp_fval'] += 1

            if opts['bb'] and bb_okay:
                # varrho_diff = varrho.ravel() - old_varrho.ravel()
                # gradient_diff = gradient.ravel() - old_gradient.ravel()
                varrho_diff = varrho - old_varrho
                gradient_diff = gradient - old_gradient
                denominator = np.vdot(gradient_diff, gradient_diff)
                if denominator > 0:
                    t = abs(np.real(np.vdot(varrho_diff, gradient_diff))) / denominator

        if i == 0 and 'boot_stats' in locals():
            boot_stats['fval'][-1] = np.real_if_close(fval_varrho)

        if t is None:
            # probs_gradient = qmt(-gradient, operators[fmap])
            probs_gradient = qst1(-gradient, nQubits)
            probs_gradient = np.concatenate((0.5*(probs_gradient[0] + probs_gradient[1:]), 0.5*(probs_gradient[0]-probs_gradient[1:])))
            # probs_gradient = probs_gradient[fmap]
            first_deriv = np.sum(-y * (probs_gradient / probs_varrho))
            second_deriv = np.sum(y * (probs_gradient ** 2 / probs_varrho ** 2))
            stats['comp_prob'] += 1
            t = -first_deriv / second_deriv
    
        else:
            if not (opts['bb'] and bb_okay):
                t *= opts['afactor']
              

        t_good = False
        fval_new = None
        t_threshold = opts['minimum_t'] / np.linalg.norm(gradient.ravel())
        while not t_good:
            if fval_new is not None:
                new_t_estimate = second_order / max(0, fval_new - fval_varrho - first_order)
                if new_t_estimate > t_threshold:
                    t = min(t * opts['bfactor'], new_t_estimate)
             
                else:
                    t *= opts['bfactor']
    

            rho_new = proj_spectrahedron(varrho - t * gradient)
            stats['comp_proj'] += 1
            # probs_rho_new = qmt(rho_new, operators[fmap])
            probs_rho_new = qst1(rho_new, nQubits)
            probs_rho_new = np.concatenate((0.5*(probs_rho_new[0] + probs_rho_new[1:]), 0.5*(probs_rho_new[0]-probs_rho_new[1:])))
            # probs_rho_new = probs_rho_new[fmap]
            fval_new = np.real_if_close(-np.sum(y * np.log(probs_rho_new)))
            # print(fval_new)
            stats['comp_prob'] += 1
            stats['comp_fval'] += 1
            stats['fval'][i] = fval_new
            # stats['fidelity'][i] = fidelity(rho_new, rho_true)
            stats['norm_err'][i] = norm(rho_new-rho_true)
            # stats['fval'].append(fval_new)
            # delta = rho_new.ravel() - varrho.ravel()
            delta = rho_new- varrho
            first_order = np.real(np.vdot(gradient, delta))
            second_order = 0.5 * np.vdot(delta, delta)
            t_good = not (t * fval_new > t * fval_varrho + t * first_order + 0.9 * second_order)

        # if fval_new < stats['best_fval']:
        #     stats['best_fval'] = fval_new
        #     stats['best_rho'] = rho_new

        if opts['save_rhos']:
            stats['rho'] = rho_new

        stats['ts'][i] = np.real_if_close(t)
        stats['thetas'][i] = theta

        stats['steps'][i] = 0.5 * np.sqrt(d) * np.linalg.norm(rho_new - rho, 'fro')
        stats['satisfied_step'] = stats['steps'][i] <= opts['threshold_step']
        stats['satisfied_fval'] = fval_new <= opts['threshold_fval']

        if opts['rho_star'] is not None:
            stats['dists'][i] = 0.5 * np.linalg.norm(rho_new - opts['rho_star'], 'fro')
            if stats['dists'][i] <= opts['threshold_dist']:
                stats['dists'][i] = 0.5 * np.sum(np.linalg.svd(rho_new - opts['rho_star'])[1])
                stats['dists_true'][i] = True
                stats['satisfied_dist'] = stats['dists'][i] <= opts['threshold_dist']

        if t < t_threshold or stats['satisfied_step'] or stats['satisfied_fval'] or stats['satisfied_dist']:
            # print(t < t_threshold, stats['satisfied_step'], stats['satisfied_fval'], stats['satisfied_dist'])
            stats['elapsed_time'][i] = time() - start
            rho = rho_new
            # print('break')
            break

        if opts['bb']:
            old_varrho = varrho.copy()

        # vec1 = varrho.ravel() - rho_new.ravel()
        # vec2 = rho_new.ravel() - rho.ravel()
        vec1 = varrho - rho_new
        vec2 = rho_new - rho
        do_restart = np.real(np.vdot(vec1, vec2)) > -opts['restart_grad_param'] * np.linalg.norm(vec1) * np.linalg.norm(vec2)

        bb_okay = not do_restart

        if do_restart:
            varrho = rho.copy()
            probs_varrho = probs_rho.copy()
            varrho_changed = theta > 1
            theta = 1
            stats['elapsed_time'][i] = time() - start
            # stats['elapsed_time'].append(time.time() - start)
            stats['fval'][i] = fval
            # stats['fidelity'][i] = fidelity(rho_new, rho_true)
            stats['norm_err'][i] = norm(rho_new- rho_true)
            if opts['save_rhos']:
                stats['rho'] = rho
            continue

        # Acceleration
        if i > 1 and stats['ts'][i] > np.finfo(float).eps:
            Lfactor = stats['ts'][i-1] / stats['ts'][i]
        else:
            Lfactor = 1

        if opts['accel'] == 'fista':
            theta_new = (1 + np.sqrt(1 + 4 * theta**2)) / 2
            beta = (theta - 1) / theta_new
            theta = theta_new
        elif opts['accel'] == 'fista_tfocs':
            theta_hat = np.sqrt(Lfactor) * theta
            theta_new = (1 + np.sqrt(1 + 4 * theta_hat**2)) / 2
            beta = (theta_hat - 1) / theta_new
            theta = theta_new
        elif opts['accel'] == 'none':
            beta = 0
        else:
            raise ValueError('Unknown acceleration scheme: {}'.format(opts['accel']))

        # Update
        varrho = rho_new + beta * (rho_new - rho)
        probs_varrho = probs_rho_new + beta * (probs_rho_new - probs_rho)
        if opts['guard'] and min(probs_varrho) <= 0:
            # Discard momentum if it causes varrho to become infeasible
            # Retain theta to keep estimate of current condition number
            varrho = rho_new
            probs_varrho = probs_rho_new

        varrho_changed = True
        rho = rho_new
        probs_rho = probs_rho_new
        fval = fval_new
        stats['elapsed_time'][i] = time() - start
        total_elapsed_time = boot_stats['elapsed_time'][-1] + stats['elapsed_time'][i]
        if total_elapsed_time >= timeLimit:
            print(f"Time limit exceeded: {total_elapsed_time:.2f}s > {timeLimit:.2f}s")
            break

    # Collect stats
    stats['n_epoch'] = stats['n_epoch'][:i+1]
    stats['fval'] = stats['fval'][:i+1]
    # stats['fidelity'] = stats['fidelity'][:i+1]
    stats['norm_err'] = stats['norm_err'][:i+1]
    stats['steps'] = stats['steps'][:i+1]
    stats['dists'] = stats['dists'][:i+1]
    stats['dists_true'] = stats['dists_true'][:i+1]
    stats['elapsed_time'] = stats['elapsed_time'][:i+1]
    stats['thetas'] = stats['thetas'][:i+1]
    stats['ts'] = stats['ts'][:i+1]

    if 'boot_stats' in locals():
        # if opts['save_rhos']:
        #     stats['rhos'] = np.concatenate((boot_stats['rhos'], stats['rhos']))
        stats['fval'] = np.concatenate((boot_stats['fval'], stats['fval']))
        stats['elapsed_time'] = np.concatenate((boot_stats['elapsed_time'], stats['elapsed_time']))
        # stats['fidelity'] = np.concatenate((boot_stats['fidelity'],  stats['fidelity']))
        stats['norm_err'] = np.concatenate((boot_stats['norm_err'],  stats['norm_err']))
        stats['n_epoch'] = np.concatenate((boot_stats['n_epoch'],  boot_stats['n_epoch'][-1] + 1+ stats['n_epoch']))
    

        # stats['steps'] = np.concatenate((boot_stats['steps'], stats['steps']))
        # stats['dists'] = np.concatenate((boot_stats['dists'], stats['dists']))
        # stats['dists_true'] = np.concatenate((boot_stats['dists_true'], stats['dists_true']))
        # stats['times'] = np.concatenate((boot_stats['times'], stats['times']))
        # stats['comp_prob'] += boot_stats['comp_prob']
        # stats['comp_grad'] += boot_stats['comp_grad']
        # stats['comp_fval'] += boot_stats['comp_fval']
        # stats['comp_proj'] += boot_stats['comp_proj']

        keys = ["n_epoch", "fidelity", "fval", "elapsed_time", 'rho', 'norm_err']
    return {k: stats[k] for k in keys if k in stats}


def qse_cgls(opts=None, **param):
    """
    QSE_CGLS Quantum state estimation via conjugate gradients with line search
    """

    name = "CG"
    print(name + " starts.")

    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y']
    nQubits = param['nQubits']
    timeLimit = param['timeLimit']

    d = 2**nQubits
    K = 2*(d**2 - 1)    
    m = 4**nQubits - 1 
    # assert K == len(f)

    defaults = {
        'threshold_step': 0.5 * np.sqrt(d) * d * np.finfo(float).eps,
        'threshold_fval': -np.sum(y[y != 0] * np.log(y[y != 0])),
        # 'threshold_fval': -np.sum(y[y != 0] * np.log((m)*y[y != 0])),
        # 'threshold_fval': -np.sum(y * np.log(y)),
        'threshold_dist': 0,
        'rho_star': None,
        'imax': 20000,
        'rho0': None,
        'save_rhos': False,
        'adjustment': 0.5,
        'mincondchange': -np.inf,
        'step_adjust': 2,
        'a2': 0.1,
        'a3': 0.2
    }

    if opts is None:
        opts = defaults
    else:
        for key, value in defaults.items():
            if key not in opts:
                opts[key] = value
    if opts['rho0'] is not None:
        V, D = np.linalg.eigh(opts['rho0'])
        temp = np.dot(np.diag(np.sqrt(np.maximum(0, np.diag(D)))), V.T)
        A = temp / np.linalg.norm(temp, 'fro')
        rho = A.T.conj()@ A
    else:
        A = np.eye(d) / np.sqrt(d)
        rho = np.eye(d) / d

    a2 = opts['a2']
    a3 = opts['a3']

    stats = {
        'steps': np.zeros(opts['imax']),
        'fval': np.zeros(opts['imax']),
        'dists': np.zeros(opts['imax']),
        'dists_true': np.zeros(opts['imax'], dtype=bool),
        'comp_prob': 0,
        'comp_grad': 0,
        'comp_fval': 0,
        'comp_proj': 0,
        'elapsed_time': np.zeros(opts['imax']),
        'satisfied_step': False,
        'satisfied_fval': False,
        'satisfied_dist': False,
        'best_rho': None,
        'best_fval': np.inf,
        'fidelity': np.zeros(opts['imax']), 
        'norm_err' : np.zeros(opts['imax']),
        'n_epoch': np.zeros(opts['imax'])
    }

    if opts['save_rhos']:
        stats['rhos'] = []

    start = time()

    probs = qst1(rho, nQubits)
    probs = np.concatenate((0.5*(probs[0] + probs[1:]), 0.5*(probs[0]-probs[1:])))
    stats['comp_prob'] += 1
    # adj = y / probs
    # # adj = y
    # adj[y == 0] = 0
    # # rmatrix = qmt(adj, operators, 'adjoint')
    rmatrix = -gradf(rho, nQubits, y, primitive1 = qst1, primitive2 = qst2)

    condchange = np.inf

    if opts['mincondchange'] > 0:
        hessian_proxy = y / (probs) ** 2
        # hessian_proxy[y == 0] = 0

    stats['comp_grad'] += 1
    # fval = -np.sum(y[y != 0] * np.log(probs[y != 0]))
    fval = -np.sum(y * np.log(probs))
    # print(np.sum(probs))
    # fval = fun(rho, nQubits, f, operators)

    stats['best_rho'] = rho
    stats['best_fval'] = fval

    i=0
    for i in range(opts['imax']):
        print(i)
        stats['n_epoch'][i] = i
        curvature_too_large = False

        if opts['mincondchange'] > 0:
            if i > 0:
                condchange = np.real(
                    np.arccos(np.real(np.vdot(old_hessian_proxy, hessian_proxy)) /
                              np.linalg.norm(old_hessian_proxy.ravel()) / np.linalg.norm(hessian_proxy.ravel())))
            # stats['cond_change_angle'][i, 1] = condchange

        if i == 0:
            # G = np.dot(A, rmatrix - np.eye(d))
            G = A@(rmatrix - np.eye(d))
            H = G
        else:
            G_next = A@(rmatrix - np.eye(d))
            polakribiere = np.real(
                np.vdot(G_next, (G_next- opts['adjustment'] * G))
            ) / np.vdot(G, G)
            gamma = max(polakribiere, 0)
            H = G_next + gamma * H
            G = G_next

        A2 = A + a2 * H
        A3 = A + a3 * H
        rho2 = A2.T.conj()@ A2
        rho2 = rho2 / np.trace(rho2)
        rho3 = A3.T.conj()@A3
        rho3 = rho3 / np.trace(rho3)
        # probs2 = qst1(rho2, nQubits)
        # probs2 = np.concatenate((0.5*(probs2[0] + probs2[1:]), (0.5*probs2[0]-probs2[1:])))
        # probs3 = qst1(rho3, nQubits)
        # probs3 = np.concatenate((0.5*(probs3[0] + probs3[1:]), (0.5*probs3[0]-probs3[1:])))
        stats['comp_prob'] += 2
        l1 = fval
        # l2 = -np.sum(y[y != 0] * np.log(probs2[y != 0]))
        # l3 = -np.sum(y[y != 0] * np.log(probs3[y != 0]))
        l2 = fun(rho2, nQubits, y, primitive1 = qst1, primitive2 = qst2)
        l3 = fun(rho3, nQubits, y, primitive1 = qst1, primitive2 = qst2)
        stats['comp_fval'] += 2
        alphaprod = 0.5 * ((l3 - l1) * a2 ** 2 - (l2 - l1) * a3 ** 2) / ((l3 - l1) * a2 - (l2 - l1) * a3)
        if np.isnan(alphaprod) or alphaprod > 1 / np.finfo(float).eps or alphaprod < 0:
            candidates = [0, a2, a3]
            index = np.argmin([l1, l2, l3])
            if opts['step_adjust'] > 1:
                if np.isnan(alphaprod) or alphaprod > 1 / np.finfo(float).eps:
                    a2 *= opts['step_adjust']
                    a3 *= opts['step_adjust']
                elif alphaprod < 0:
                    a2 /= opts['step_adjust']
                    a3 /= opts['step_adjust']
                    curvature_too_large = True
            alphaprod = candidates[index]

        A = A + alphaprod * H
        A = A / np.linalg.norm(A, 'fro')
        old_rho = rho
        rho = A.T.conj()@ A

        if opts['save_rhos']:
            stats['rhos'].append(rho)

        # stats['alphas'][i] = alphaprod

        # probs = qst1(rho, nQubits)
        stats['comp_prob'] += 1
        fval = fun(rho, nQubits, y, primitive1 = qst1, primitive2 = qst2)
        stats['elapsed_time'][i] = time() - start
        stats['comp_fval'] += 1
        stats['fval'][i] = np.real_if_close(fval)
        stats['norm_err'][i] = norm(rho-rho_true)
        # stats['fidelity'][i] = fidelity(rho, rho_true)

        # if stats['fval'][i] < stats['best_fval']:
        #     stats['best_fval'] = stats['fval'][i]
        #     stats['best_rho'] = rho

        if stats['elapsed_time'][i] >= timeLimit:
            print(f"Time limit exceeded: {stats['elapsed_time'][i]:.2f}s > {timeLimit:.2f}s")
            break

        stats['steps'][i] = 0.5 * np.sqrt(d) * np.linalg.norm(rho - old_rho, 'fro')
        stats['satisfied_step'] = stats['steps'][i] <= opts['threshold_step']
        # print(fval)
        stats['satisfied_fval'] = fval <= opts['threshold_fval']
        if opts['rho_star'] is not None:
            stats['dists'][i] = 0.5 * np.linalg.norm(rho - opts['rho_star'], 'fro')
            if stats['dists'][i] <= opts['threshold_dist']:
                stats['dists'][i] = 0.5 * np.sum(np.linalg.svd(rho - opts['rho_star'])[1])
                stats['dists_true'][i] = True
                stats['satisfied_dist'] = stats['dists'][i] <= opts['threshold_dist']

        if (not curvature_too_large and stats['satisfied_step']) or stats['satisfied_fval'] or condchange < opts['mincondchange'] or stats['satisfied_dist']:
            break
        
        if i != opts['imax'] - 1:
            # adj = y / probs
            # # adj = y
            # adj[y == 0] = 0
            rmatrix = -grad(rho, nQubits, y, primitive1 = qst1, primitive2 = qst2)
            stats['comp_grad'] += 1
            if opts['mincondchange'] > 0:
                old_hessian_proxy = hessian_proxy
                hessian_proxy = y / (probs) ** 2
                # hessian_proxy[y == 0] = 0

    stats['n_epoch'] = stats['n_epoch'][:i+1]
    stats['fval'] = stats['fval'][:i+1]
    # stats['fidelity'] = stats['fidelity'][:i+1]
    stats['norm_err'] = stats['norm_err'][:i+1]
    stats['steps'] = stats['steps'][:i+1]
    stats['dists'] = stats['dists'][:i+1]
    stats['dists_true'] = stats['dists_true'][:i+1]
    stats['elapsed_time'] = stats['elapsed_time'][:i+1]
    return rho, stats
'''


