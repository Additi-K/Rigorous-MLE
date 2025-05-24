# import libraries

import numpy as np
import scipy
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from time import time
# from proxop import Simplex
from functools import partial
from utils import *
import cvxpy as cp
import sparse
from firstOrderMethods import *


# Helper functions
def init_output(n_epoch):
    output = {}
    output["n_epoch"]      = []
    output["fidelity"]     = []
    output["fval"]         = []
    output["elapsed_time"] = []
    output["norm_err"] = []

    return output
    
def update_output(output, t, fid, fval, tme, err):
    output["fidelity"].append(fid)
    output["fval"].append(fval)
    output["n_epoch"].append(t)
    output["elapsed_time"].append(tme)
    output["norm_err"].append(err)

    return output

def dot(x,y):
    return np.real(np.dot(x.conj().reshape(-1, ),y.reshape(-1, )))


def norm(x):
     return np.linalg.norm(x, 'fro')


def my_eigsh(X,r):
    n = np.shape(X)[0]
    r = np.minimum(n, r)
    if r > n/2:
        # Do a dense eigenvalue solve
        # eigvals, eigvecs = eigh(X,subset_by_index=[n-r,n-1])  # Stephen 
        eigvals, eigvecs = eigh(X,subset_by_index=[n-r -1 ,n-1])
    else:
        # eigvals, eigvecs = eigsh(X, k = r, which = 'LM')  # Stephen
        eigvals, eigvecs = eigsh(X, k = r+1, which = 'LM')
    
    return eigvals, eigvecs, r


# Algorithms 

##########################################
# Exponentiated mirror descent
##########################################
def EMD(**param):

    # read the required parameters
    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y']
    nQubits = param['nQubits']
    timeLimit = param['timeLimit']
    n_epoch = 1000
    alpha0 = 10.0
    r = 0.5
    lmbda = 0.5

    name = "EMD"
    print(name + " starts.")
    out_emd = init_output(n_epoch)
    
    fval = np.zeros((1), dtype= np.complex64)
    d = rho_true.shape[0]
    
    x = np.eye(d, dtype= np.complex64) / d
    
    grad = np.zeros((d,d), dtype= np.complex64)
    fval = fun(x, nQubits, y, primitive1 = qst1, primitive2 = qst2)
    
    start_time = time()
    # Main loop
    for t in range(0, n_epoch ):
        if t%20 == 0:
            print("iteration", t)
        
        grad = gradf(x, nQubits, y, primitive1 = qst1, primitive2 = qst2)

        # Armijo line search
        alpha = alpha0
        x_alpha = scipy.linalg.expm(scipy.linalg.logm(x) - alpha * grad)
        x_alpha /= np.trace(x_alpha)

        round = 0
        while lmbda  * dot(grad, x_alpha - x) + fval < fun(x_alpha, nQubits ,y,primitive1 = qst1, primitive2 = qst2) and round < 10:
            alpha *= r
            x_alpha = scipy.linalg.expm(scipy.linalg.logm(x) - alpha * grad)
            x_alpha /= np.trace(x_alpha)
            round += 1
        
        if round < 10:
            x = x_alpha
        else:
            x = x   

        fval = fun(x, nQubits, y, primitive1 = qst1, primitive2 = qst2)
        cum_time = time() - start_time

        # Update output
        update_output(out_emd, t, fidelity(x, rho_true), fval, cum_time, norm(x-rho_true))
        if  out_emd['elapsed_time'][-1] >= timeLimit:
            print(f"Time limit exceeded: {out_emd['elapsed_time'][-1]:.2f}s > {timeLimit:.2f}s")
            break
        
    out_emd['rhoEst'] = x     
    
    return out_emd



####################################################
# Approximated Exponentiated gradient descent method
####################################################

def approx_MEG(**param):    
    """
    eficient implementation of matrix exponentiated gradient descent
    
    ref: On the Efficient Implementation of the Matrix Exponentiated Gradient Algorithm 
           for Low-Rank Matrix Optimization, Dan Garber, Atara Kaplan
    
    input is matrix
    """

    # read the required parameters
    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y']
    r_svd = param['r_svd']
    warmstart = param['warmstart']
    eta = param['eta']
    eps_coeff = param['eps_coeff']
    c_eps = param['c_eps']
    nQubits = param['nQubits']
    timeLimit = param['timeLimit']
    max_iter = 1000
    n = 2**nQubits


    name = "Approximate MEG"
    print(name + " starts.")

    X1 = initialize_MEG(warmstart, n, r_svd, gradf, eps_coeff, c_eps, y)
    out_meg = lowRankMirrorDescent(X1, rho_true, n, r_svd, fun, gradf, eta, c_eps, eps_coeff, max_iter, y, timeLimit)

    return out_meg 

def initialize_MEG(warmStart, n, r_svd, gradf, eps_coeff, c_eps, y):
    tau = 0.5             #tr(X) = tau, tau= 0.5*tr(X) in our case
    if warmStart == 1:
        x = np.random.normal(size=(n, r_svd))
        x = x / np.linalg.norm(x, 'fro')
        grad = gradf(x @ x.T.conj(), int(np.log2(n)),y, primitive1 = qst1, primitive2 = qst2)
        grad = (grad + grad.conj().T)/2
        eigvals, eigvecs, _ = my_eigsh(grad,r=r_svd)
        eig_project = Simplex(eta=tau).prox(eigvals) #Simplex(eigvals, tau)
        X1_tilde = eigvecs @ np.diag(eig_project) @ eigvecs.T.conj()
        eps = eps_coeff * 0.75 / (c_eps + 1)**2
        X1 = (1 - eps) * X1_tilde + tau * (eps / n) * np.eye(n)
    else:
        x = np.random.normal(size=(n, r_svd))
        x = x / np.linalg.norm(x, 'fro')
        X1_tilde = tau * (x @ x.T)
        eps = eps_coeff * 0.75 / (c_eps + 1)**2
        X1 = (1 - eps) * X1_tilde + tau * (eps / n) * np.eye(n)
    return X1    

def lowRankMirrorDescent(X, rhoTrue, n, r_svd, fun, gradf, eta, c_eps, eps_coeff, max_iter, y, timeLimit):
    tau = 0.5
    out_meg = init_output(max_iter)
    start_time = time()
    for i in range(0, max_iter):
        if i%20 == 0:
            print("iteration", i)
        eps = eps_coeff * 3/4 /(i + c_eps + 1 + 1)**2  # (4/5 in paper)
        grad = gradf(X , int(np.log2(n)), y, primitive1 = qst1, primitive2 = qst2)
        grad = (grad + grad.conj().T)/2
        Y = scipy.linalg.expm(scipy.linalg.logm(X) - eta * grad)
        # Y = X*scipy.linalg.expm(-eta * grad)

        eigvals, eigvecs, r_exp = my_eigsh(Y,r=r_svd)

        bt = np.trace(Y)
        Yr = eigvecs[:, 0:r_svd]@np.diag(eigvals[0:r_svd])@eigvecs[:, 0:r_svd].T.conj()
        at = np.trace(Yr)
        # print(f'{bt=}, {at=}')
        X = tau*((1-eps)/ at) * Yr + tau*(eps/(n-r_svd)) * (np.eye(n) -  eigvecs[:, 0:r_svd]@eigvecs[:, 0:r_svd].T.conj())
        fval = fun(X , int(np.log2(n)), y, primitive1 = qst1, primitive2 = qst2)
        
        cum_time = time() - start_time
        # Update output
        update_output(out_meg, i,  fidelity(X, rhoTrue), fval, cum_time, norm(X-rhoTrue)) 
        if  out_meg['elapsed_time'][-1] >= timeLimit:
            print(f"Time limit exceeded: {out_meg['elapsed_time'][-1]:.2f}s > {timeLimit:.2f}s")
            break

    out_meg['rhoEst'] = X
    out_meg['r_exp'] = r_exp

    return out_meg   


######################################################
## Stochastic dual averaging with logarithmic barrier
######################################################
def alpha(rho, v):
    # print(-np.vdot(rho, (rho @ v).T.conj())/np.vdot(rho.conj().T, rho),  -np.real(np.trace(rho @ v @ rho) / np.trace(rho @ rho)))
    return -np.real_if_close(np.vdot(rho, (rho @ v).T.conj())/np.vdot(rho.conj().T, rho))


def dual_norm2(rho, sig):
    prod = rho @ sig
    return np.real_if_close(np.vdot(prod.conj().T, prod))
   
def log_barrier_projection(u, epsilon):
    theta = 1 - np.min(1.0 / u)
    a = 1.0 / ((1.0 / u) + theta)
    grad = 1 - np.sum(a)
    grad2 = np.vdot(a, a)
    lmbda_t = abs(grad) / np.sqrt(grad2)

    while lmbda_t > epsilon:
        a = 1.0 / ((1.0 / u) + theta)
        grad = 1 - np.linalg.norm(a, 1)
        grad2 = np.vdot(a, a)
        theta = theta - grad / grad2
        lmbda_t = abs(grad) / np.sqrt(grad2)

    return 1.0 / ((1.0 / u) + theta) 


def LBSDA(**param):
    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y']
    A = param['A'].todense()
    n_epoch = param['max_iter1']
    n_rate = param['n_rate']
    nQubits = param['nQubits']
    timeLimit = param['timeLimit']

    name = "1-sample LB-SDA"
    print(name + " starts.")
    
    d = rho_true.shape[0]
    nShots = 4**nQubits*100

    rhoBar = np.eye(d, dtype=np.complex128) / d
    rho = np.eye(d, dtype=np.complex128) / d
    sum_grad = np.zeros((d, d), dtype=np.complex128)
    grad = np.zeros((d, d), dtype=np.complex64)
    sum_dual_norm2 = 0.0
    n_iter = n_epoch * nShots
    out_1_lbsda = init_output(n_epoch)
    period = nShots // n_rate
    idx = np.random.randint(0, 2* (4**nQubits-1), n_iter)

    # cnt = 0
    # print(cnt)

    start_time = time()
    # print(n_iter, period)
    for iter in range(0, n_iter):
        y_sample = y[idx[iter]:idx[iter]+1]
        A_sample= A[idx[iter]:idx[iter]+1, :, :].squeeze()
        grad = -y_sample *A_sample/np.vdot(A_sample, rhoBar)
        # grad = gradf(rhoBar, nQubits, y_sample, primitive1 = qst1, primitive2 = qst2, A = A_sample)
        sum_dual_norm2 += dual_norm2(rho, grad + alpha(rho, grad) * np.eye(d))
        eta = np.sqrt(d) / np.sqrt(4 * d + 1 + sum_dual_norm2)
        sum_grad += grad
        Lmbda_inv, U = eigh(eta * sum_grad)
        Lmbda = log_barrier_projection(1.0 / Lmbda_inv, 1e-5)
        rho = U @ np.diag(Lmbda) @ U.conj().T
        rhoBar = ((iter+1.0) * rhoBar + rho) / ((iter + 1.0) + 1.0)

        curr_time = time()
        if  (curr_time - start_time) >= timeLimit:
            print(f"Time limit exceeded: {curr_time - start_time:.2f}s > {timeLimit:.2f}s")
            update_output(out_1_lbsda, out_1_lbsda['n_epoch'][-1]+1 , fidelity(rhoBar, rho_true), fun(rhoBar , nQubits, y, qst1), curr_time - start_time, norm(rhoBar-rho_true))
            break 

        if (iter+1) % period == 0:
            print((iter+1) / period)
            funval = fun(rhoBar , nQubits, y, primitive1 = qst1, primitive2 = qst2)
            end_time = time()
            update_output(out_1_lbsda, iter // period, fidelity(rhoBar, rho_true), funval, end_time - start_time, norm(rhoBar-rho_true))
            # cnt = cnt + 1
            # if cnt%10 ==0:
            #     print(cnt) 


    out_1_lbsda['rho'] = rhoBar          
    return out_1_lbsda


def d_LBSDA(**param):
    
    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y']
    A = param['A']
    n_epoch = param['max_iter2']
    n_rate = param['n_rate']
    nQubits = param['nQubits']
    timeLimit = param['timeLimit']
    
    name = "d-sample LB-SDA"
    print(name + " starts.")

    d = rho_true.shape[0]
    nShots = 4**nQubits*100
    rhoBar = np.eye(d, dtype=np.complex128) / d
    rho = np.eye(d, dtype=np.complex128) / d
    sum_grad = np.zeros((d, d), dtype=np.complex128)
    sum_dual_norm2 = 0.0
    batch_size = d

    n_iter = n_epoch * int(nShots/batch_size)
    period = nShots // n_rate // batch_size
    out_d_lbsda = init_output(n_iter)
    idx = np.random.randint(0, 2* (4**nQubits-1), size=(batch_size, n_iter))

    # cnt = 0
    # print(cnt)

    start_time = time()
    for iter in range(0, n_iter):
        grad = np.zeros((d, d), dtype=np.complex64)
        grad = gradf(rhoBar, nQubits, y[idx[:, iter]], primitive1 = qst1, primitive2 = qst2, A =  A[idx[:, iter]])#.todense()
        grad /= batch_size    
        sum_grad += grad
        sum_dual_norm2 += dual_norm2(rho, grad + alpha(rho, grad) * np.eye(d, dtype=np.complex128))
        eta = np.sqrt(d) / np.sqrt(4 * d + 1 + sum_dual_norm2)
        
        Lmbda_inv, U = eigh(eta * sum_grad)
        # Lmbda_inv, U = eigsh(eta * sum_grad , k = sum_grad.shape[0]-2)
        # Lmbda_inv, U = eig(eta * sum_grad)
        Lmbda = log_barrier_projection(1.0 / Lmbda_inv, 1e-5)
        rho = U @ np.diag(Lmbda) @ U.conj().T
        rhoBar = ((iter+1.0) * rhoBar + rho) / ((iter + 1.0) + 1.0)
        curr_time = time()
        if  (curr_time - start_time) >= timeLimit:
            print(f"Time limit exceeded: {curr_time - start_time:.2f}s > {timeLimit:.2f}s")
            update_output(out_d_lbsda, out_d_lbsda['n_epoch'][-1]+1 , fidelity(rhoBar, rho_true), fun(rhoBar , nQubits, y, primitive1 = qst1, primitive2 = qst2), curr_time - start_time, norm(rhoBar-rho_true))
            break 


        if (iter+1) % period == 0:
            print((iter+1) / period)
            funval = fun(rhoBar , nQubits, y,  primitive1 = qst1, primitive2 = qst2)
            end_time = time()
            update_output(out_d_lbsda, iter // period, fidelity(rhoBar, rho_true), funval, end_time - start_time, norm(rhoBar-rho_true))
            # cnt = cnt + 1
            # if (cnt+1)%10 ==0:
            #     print(cnt) 

    out_d_lbsda['rho'] = rhoBar
    return out_d_lbsda


class TimeExceededException(Exception):
    """Custom exception to stop optimization when time limit is exceeded."""
    pass

class TimingCallback:
    def __init__(self, rhoTrue, fidelity, get_rho, fun, gradf, y, nQubits, rank, timeLimit):
        self.elapsed_time = []
        self.fval = []
        self.xval = []
        self.fidelity = []
        self.rhoTrue = rhoTrue
        self.fidelity_fun = fidelity
        self.get_rho_fun = get_rho
        self.nQubits = nQubits
        self.rank = rank
        self.cnt = 0
        self.n_epoch = []
        self.gradf = gradf
        self.fun = fun
        self.y = y
        self.timeLimit = timeLimit
        self.norm_err = []
        self.duality_gap = []
        self.start_time = time()


    def __call__(self, intermediate_result):
 
        # update stats
        
        self.xval = np.copy(intermediate_result.x)
        self.n_epoch.append(self.cnt)
        self.cnt += 1
        
        # Calculate and store fidelity
        rhoest, u = self.get_rho_fun(intermediate_result.x, self.nQubits, self.rank)
        rhoest = rhoest/np.trace(rhoest)
        fval = self.fun(rhoest, self.nQubits, self.y, primitive1 = qst1, primitive2=qst2)
        # print(fval)
        self.fval.append(fval)
        fidval = self.fidelity_fun(rhoest, self.rhoTrue)
        self.fidelity.append(fidval)
        self.elapsed_time.append(time() - self.start_time)
        self.norm_err.append(norm(rhoest-self.rhoTrue))

        
        if self.cnt%20 == 0:
            print(self.cnt)
        
        # check optimality cond <rho, alpha> ==0
        prob = qst1(rhoest, self.nQubits)
        prob = np.concatenate([0.5*(prob[0]+ prob[1:]), 0.5*(prob[0]- prob[1:])])
        eta = np.sum(self.y/prob)

        gradient = self.gradf(rhoest, self.nQubits, self.y, primitive1= qst1, primitive2=qst2)
        B = eta*np.eye(2**self.nQubits) - gradient
        max_eig_B, _ = scipy.sparse.linalg.eigsh(B,k = 1,  which = 'LM')

        mu = (eta-max_eig_B)
        mu = scipy.sparse.linalg.eigsh(gradient, k = 1, which = 'SA')[0]
        mu = - mu 

        sigma = gradient + mu*np.eye(2**self.nQubits)
        gap = np.trace(sigma.conj().T@rhoest)
        self.duality_gap.append(gap)
            
        print(f'  <sigma,rho> = {gap:.2e}')
    
        e3 = np.min( np.linalg.eigvalsh(sigma) )
        print(f'min(eig(sigma)) is {e3:.2e}')

        # Check if total elapsed time exceeds the limit
        if self.elapsed_time[-1] > self.timeLimit:
            print(f"Time limit exceeded: {self.elapsed_time[-1]:.2f}s > {self.timeLimit:.2f}s")
            raise TimeExceededException("Time limit exceeded, stopping optimization")


def reshape(input, nQubits, rank):
    idx = int(input.shape[0]/2)
    reshaped_u = input[0:idx].reshape((2**nQubits, rank), order = 'C') + 1j*input[idx:].reshape((2**nQubits, rank), order='C')

    return reshaped_u

def get_rho(u, nQubits, rank):
    reshaped_u = reshape(u, nQubits, rank)
    rhoest = reshaped_u@np.conj(reshaped_u.T)
    
    return rhoest, reshaped_u


def LBFGS(**param):
    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y'] 
    nQubits = param['nQubits'] 
    r_svd = param['r_svd']
    timeLimit = param['timeLimit']
    
    tau = 2*np.sum(y)  #  np.sum(yPlus+yMinus) = 1, tau = 2 
    name = "LBFGS"
    print(name + " starts.")

    for i in range(1):
        dim = 2*r_svd*(2)**nQubits
        u0 = np.random.randn((dim))
        u0= u0/ np.linalg.norm(u0)


    timingcallback = TimingCallback(rho_true, fidelity, get_rho, fun, gradf, y, nQubits, r_svd, timeLimit)            
    
    try:
        LBFGS_out = scipy.optimize.minimize(fun, u0 , args=(nQubits, y, qst1, qst2, r_svd), 
                                        method='L-BFGS-B', jac=gradf, 
                                        options={'disp':False, 'ftol':1e-15,'gtol':1e-9, 
                                        'iprint':-1, 'maxiter': 2e3,  'maxfun' : 1e5, 'maxcor':2}, callback=timingcallback )
    except TimeExceededException as e:
        print(e)
        return {'u':timingcallback.xval, 'fidelity':timingcallback.fidelity, 'fval':timingcallback.fval, 
        'elapsed_time':timingcallback.elapsed_time, 'n_epoch': timingcallback.n_epoch, 'rho': get_rho(timingcallback.xval, nQubits, r_svd)[0],
        'norm_err':timingcallback.norm_err, 'gap': timingcallback.duality_gap}   

    return {'u':LBFGS_out.x, 'fidelity':timingcallback.fidelity, 'fval':timingcallback.fval, 
            'elapsed_time':timingcallback.elapsed_time, 'n_epoch': timingcallback.n_epoch,  'rho': get_rho(timingcallback.xval, nQubits, r_svd)[0], 
            'norm_err':timingcallback.norm_err, 'gap': timingcallback.duality_gap }

def cvxpy_convex(**param):
    nQubits = param['nQubits']
    if nQubits >= 8:
        return None
    rho_true = param['rho_true']
    y = param['y']
    A = param['A']
    
    solver = param['cvxSolver']
    fun = param['fun']
    
    name = "CVXPY"
    print(name + " starts.")

    cvxpy_objective = lambda rho, y, A : cp.sum(cp.multiply(-y , cp.log(cp.real(sparse.reshape(A, (-1, 4**nQubits)).todense()@cp.vec(rho)))))
    rhoCVX = cp.Variable((2**nQubits,2**nQubits), hermitian = True)
    cost = cvxpy_objective(rhoCVX, y, A)
    constraint = [rhoCVX>>0, cp.real(cp.trace(rhoCVX))==1 ]

    prob = cp.Problem(cp.Minimize(cost), constraint )
    start = time()
    tol = 1e-10
    if solver == 'ECOS':
        ECOSopts = {'abstol':tol, 'reltol':tol, 'feastol':tol, 'max_iters':int(2e4) }
        prob.solve( solver = cp.ECOS, verbose=False, **ECOSopts)
    elif solver == 'SCS':    
        SCSopts  = {'max_iters':20000, 'eps':tol}
        prob.solve(verbose=False, solver=cp.SCS, **SCSopts)
    else:
        prob.solve(solver = cp.CVXOPT) 

    rhoCVX = rhoCVX.value
    print('time cvx', time()-start)
    return rhoCVX



def AccGD(**param):

    fun = param['fun']
    gradf = param['gradf']
    y = param['y'] 
    nQubits = param['nQubits'] 
    r_svd = param['r_svd']
    timeLimit = param['timeLimit']


    dim = 2*r_svd*(2)**nQubits
    x0 = np.random.randn((dim))
    x0= x0/ np.linalg.norm(x0)

    xNew, data = gradientDescent(fun,gradf,x0,stepsize=None,
                        tol=1e-15, saveHistory=True, printEvery=100,
                        acceleration= True, maxIters=5e4, ArmijoLinesearch = False,
                        args= (nQubits, y, qst1, qst2, r_svd), timeLimit = timeLimit)
    
    return {'u':xNew, 'rho': get_rho(xNew, nQubits, r_svd)[0], 'elapsed_time': data['elapsed_time'], 
            'n_epoch':np.arange(data['steps']), 'fval': data['fcnHistory']}


def LBFGS_lowmem(**param):
    rho_true = param['rho_true']
    fun = param['fun']
    gradf = param['gradf']
    y = param['y'] 
    nQubits = param['nQubits'] 
    r_svd = param['r_svd']
    timeLimit = param['timeLimit']
    mList = param['mList']
    
    # tau = 2*np.sum(y)  #  np.sum(yPlus+yMinus) = 1, tau = 2 
    name = "LBFGS"
    print(name + " starts.")

    for i in range(1):
        dim = 2*r_svd*(2)**nQubits
        u0 = np.random.randn((dim))
        u0= u0/ np.linalg.norm(u0)


    timingcallback = TimingCallback_2(rho_true, fidelity, get_rho, lowmemA, mList, y, nQubits, r_svd, timeLimit)            
    
    try:
        LBFGS_out = scipy.optimize.minimize(lowmemA, u0 , args=(nQubits, mList, r_svd, y), 
                                        method='L-BFGS-B', jac=True, 
                                        options={'disp':False, 'ftol':1e-15,'gtol':1e-9, 
                                        'iprint':-1, 'maxiter': 10000,  'maxfun' : 1e5, 'maxcor':2}, callback=timingcallback )
    except TimeExceededException as e:
        print(e)
        return {'u':timingcallback.xval, 'fidelity':timingcallback.fidelity, 'fval':timingcallback.fval, 
        'elapsed_time':timingcallback.elapsed_time, 'n_epoch': timingcallback.n_epoch, 'rho': get_rho(timingcallback.xval, nQubits, r_svd)[0],
        'norm_err':timingcallback.norm_err, 'gap': timingcallback.duality_gap}   

    return {'u':LBFGS_out.x, 'fidelity':timingcallback.fidelity, 'fval':timingcallback.fval, 
            'elapsed_time':timingcallback.elapsed_time, 'n_epoch': timingcallback.n_epoch,  'rho': get_rho(timingcallback.xval, nQubits, r_svd)[0], 
            'norm_err':timingcallback.norm_err, 'gap': timingcallback.duality_gap }

class TimingCallback_2:
    def __init__(self, rhoTrue, fidelity, get_rho, fun_and_grad, mList, y, nQubits, rank, timeLimit):
        self.elapsed_time = []
        self.fval = []
        self.xval = []
        self.fidelity = []
        self.rhoTrue = rhoTrue
        self.fidelity_fun = fidelity
        self.get_rho_fun = get_rho
        self.nQubits = nQubits
        self.rank = rank
        self.cnt = 0
        self.n_epoch = []
        self.fun_and_grad = fun_and_grad
        self.y = y
        self.mList = mList
        self.timeLimit = timeLimit
        self.norm_err = []
        self.duality_gap = []
        self.start_time = time()


    def __call__(self, intermediate_result):
 
        # update stats
        self.elapsed_time.append(time() - self.start_time)
        self.xval = np.copy(intermediate_result.x)
        self.n_epoch.append(self.cnt)
        self.cnt += 1
        
        # Calculate and store fidelity
        rhoest , u = self.get_rho_fun(intermediate_result.x, self.nQubits, self.rank)
        rhoest = rhoest/np.trace(rhoest)
        self.fval.append(self.fun_and_grad(intermediate_result.x, self.nQubits, self.mList, self.rank, self.y)[0])
        fidval = self.fidelity_fun(rhoest, self.rhoTrue)
        self.fidelity.append(fidval)
        self.norm_err.append(norm(rhoest-self.rhoTrue))

        
        if self.cnt%5 == 0:
            print(self.cnt)
        
    
        # Check if total elapsed time exceeds the limit
        if self.elapsed_time[-1] > self.timeLimit:
            print(f"Time limit exceeded: {self.elapsed_time[-1]:.2f}s > {self.timeLimit:.2f}s")
            raise TimeExceededException("Time limit exceeded, stopping optimization")
