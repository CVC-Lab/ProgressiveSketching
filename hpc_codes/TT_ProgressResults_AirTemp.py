#################################################################################

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits import mplot3d
from numpy.random import normal, choice, rand, beta, binomial, randint, dirichlet
from numpy.linalg import matrix_rank, norm, svd, eig, qr
from scipy.linalg import pinv2
from scipy import interpolate
from scipy.stats import entropy
def pinv(x): return pinv2(x, rcond=10 * np.finfo(float).eps)
import time

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

#################################################################################
def tt_PGSVD_log(A,rank):
    d = len(A.shape)
    
    err_log = []
    for p in np.linspace(0.003,0.05,20):
        r = [1]
        G = []
        C = A
        l = [np.int(p*(A.shape[0] + A.shape[1]*A.shape[2])),np.int(p*(rank[0]*A.shape[1] + A.shape[2]))]
        for k in range(1,d):
            C = C.reshape((np.int(r[k-1]*A.shape[k-1]),np.int(C.size/r[k-1]/A.shape[k-1])),order='F')
            U, s, VT = PG_SCSVD(C, rank[k-1], 10, 10, l[k-1])
            r.append(rank[k-1])
            U = U[:, :r[k]]
            s = s[:r[k],:r[k]]
            VT = VT[:r[k], :]
            G.append(U.reshape((r[k-1],A.shape[k-1],r[k]),order='F'))
            C = np.dot(s,VT)
        G.append(C[:,:,np.newaxis])

        A_lowrank = G[0].reshape(A.shape[0], -1)
        A_lowrank = A_lowrank.dot(G[1].reshape(r[1],-1)).reshape(-1,r[2])
        A_lowrank = A_lowrank.dot(G[2].reshape(r[2],-1)).reshape(A.shape[0],A.shape[1],A.shape[2])

        err_log.append(norm(A_lowrank - A)**2 / norm(A)**2)

    return err_log
    
def tt_RDSVD_log(A,rank):
    d = len(A.shape)
    
    err_log = []
    for p in np.linspace(0.003,0.05,20):
        r = [1]
        G = []
        C = A
        l = [np.int(p*(A.shape[0] + A.shape[1]*A.shape[2])),np.int(p*(rank[0]*A.shape[1] + A.shape[2]))]
        for k in range(1,d):
            C = C.reshape((np.int(r[k-1]*A.shape[k-1]),np.int(C.size/r[k-1]/A.shape[k-1])),order='F')
            U, s, VT = RD_SCSVD(C, rank[k-1], 10, 10, l[k-1])
            r.append(rank[k-1])
            U = U[:, :r[k]]
            s = s[:r[k],:r[k]]
            VT = VT[:r[k], :]
            G.append(U.reshape((r[k-1],A.shape[k-1],r[k]),order='F'))
            C = np.dot(s,VT)
        G.append(C[:,:,np.newaxis])

        A_lowrank = G[0].reshape(A.shape[0], -1)
        A_lowrank = A_lowrank.dot(G[1].reshape(r[1],-1)).reshape(-1,r[2])
        A_lowrank = A_lowrank.dot(G[2].reshape(r[2],-1)).reshape(A.shape[0],A.shape[1],A.shape[2])

        err_log.append(norm(A_lowrank - A)**2 / norm(A)**2)

    return err_log

mat = scipy.io.loadmat('Cardiac_DE_2.mat')
A = mat['X']
N = A.shape
r = np.array([20,20,5])

# matrix unfoldings
A0 = A.reshape(N[0],N[1]*N[2]) # 0 1 2
tmp = np.moveaxis(A,0,2) # 1 2 0
A1 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])
tmp = np.moveaxis(A,2,0) # 2 0 1
A2 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])

for i in tqdm(range(10)):
    logPS_0, logPS_1, logPS_2 = pgskt_log(A,r,10,10,500)
    logRS_0, logRS_1, logRS_2 = rdskt_log(A,r,10,10,500)
    err_logPS = lc(N,logPS_0, logPS_1, logPS_2)
    err_logRS = lc(N,logRS_0, logRS_1, logRS_2)
    nf_PS = [len(i)+len(j)+len(k) for i,j,k in zip(logPS_0,logPS_1,logPS_2)]
    nf_RS = [len(i)+len(j)+len(k) for i,j,k in zip(logRS_0,logRS_1,logRS_2)]
    np.save(f'outputs/Cardiac_err_logPS_{i}.npy',np.array(err_logPS))
    np.save(f'outputs/Cardiac_err_logRS_{i}.npy',np.array(err_logRS))
    np.save(f'outputs/Cardiac_nf_PS_{i}.npy',np.array(nf_PS))
    np.save(f'outputs/Cardiac_nf_RS_{i}.npy',np.array(nf_RS))
