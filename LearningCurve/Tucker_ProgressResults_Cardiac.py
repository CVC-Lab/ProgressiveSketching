#################################################################################

import numpy as np
from netCDF4 import Dataset
import scipy.io
from numpy.random import normal, choice, rand, beta, binomial, randint, dirichlet
from numpy.linalg import matrix_rank, norm, svd, eig, qr
from scipy.linalg import pinv
from scipy import interpolate
from scipy.stats import entropy
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#################################################################################
def HOSVD(A):
    A1 = A.reshape(A.shape[0],A.shape[1]*A.shape[2]) # 1 2 3
    tmp = np.moveaxis(A,0,2) # 2 3 1
    A2 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])
    tmp = np.moveaxis(A,2,0) # 3 1 2
    A3 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])
    U1, s1, V1 = svd(A1,full_matrices=False)
    U2, s2, V2 = svd(A2,full_matrices=False)
    U3, s3, V3 = svd(A3,full_matrices=False)
    
    S = np.tensordot(np.tensordot(np.tensordot(A,U1.T,axes=(0,1)),
                              U2.T,axes=(0,1)),
                 U3.T,axes=(0,1))
    return S, U1, U2, U3

def linear(index,score,N):
    f = interpolate.interp1d(index, score)
    z = f(np.arange(N))
    z[index] = 0
    if np.sum(z) > 0:
        return z / np.sum(z)
    if np.sum(z) == 0:
        return None

def update_z(index,N):
    z = np.ones(N)
    z[index] = 0
    if np.sum(z) > 0:
        return z / np.sum(z)
    if np.sum(z) == 0:
        return None
    
def pgskt_log(A,r,start,step,stop):
    N = np.array(A.shape)
    m = np.array([start,start,start],dtype=np.int)
    params = [1,1,1]
    
    # inital core uniform sampling
    index_0 = np.append(choice(N[0]-2,start-2,replace=False)+1,[0,N[0]-1])
    index_1 = np.append(choice(N[1]-2,start-2,replace=False)+1,[0,N[1]-1])
    index_2 = np.append(choice(N[2]-2,start-2,replace=False)+1,[0,N[2]-1])
    index_0_log = [index_0.tolist()]
    index_1_log = [index_1.tolist()]
    index_2_log = [index_2.tolist()]
    
    # order 2 SAD
    score_0 = np.sum(np.sum(np.abs(np.diff(A[index_0,:,:],axis=1)),axis=1)/N[1]/N[2],axis=1) + np.sum(np.sum(np.abs(np.diff(A[index_0,:,:],axis=2)),axis=2)/N[2]/N[1],axis=1)
    score_1 = np.sum(np.sum(np.abs(np.diff(A[:,index_1,:],axis=2)),axis=2)/N[2]/N[0],axis=0) + np.sum(np.sum(np.abs(np.diff(A[:,index_1,:],axis=0)),axis=0)/N[0]/N[2],axis=1)
    score_2 = np.sum(np.sum(np.abs(np.diff(A[:,:,index_2],axis=0)),axis=0)/N[0]/N[1],axis=0) + np.sum(np.sum(np.abs(np.diff(A[:,:,index_2],axis=1)),axis=1)/N[1]/N[0],axis=0)
    
    # bilinear interpolation
    z_0 = linear(index_0,score_0,N[0])
    z_1 = linear(index_1,score_1,N[1])
    z_2 = linear(index_2,score_2,N[2])
    
    while np.sum(m) < stop:
        theta = np.array(dirichlet(params,1)[0] * step,dtype=np.int)
        
        if z_0 is not None:
            if np.sum(z_0>0) > theta[0]:
                add = theta[0]
            else:
                add = np.sum(z_0>0)
            m[0] += add
            new_index = choice(N[0],add,replace=False,p=z_0)
            new_score = np.sum(np.sum(np.abs(np.diff(A[new_index,:,:],axis=1)),axis=1)/N[1]/N[2],axis=1) + np.sum(np.sum(np.abs(np.diff(A[new_index,:,:],axis=2)),axis=2)/N[2]/N[1],axis=1)
            index_0 = np.append(index_0,new_index)
            score_0 = np.append(score_0,new_score)
            z_0 = linear(index_0,score_0,N[0])
            params[0] = entropy(score_0)
            
        if z_1 is not None:
            if np.sum(z_1>0) > theta[1]:
                add = theta[1]
            else:
                add = np.sum(z_1>0)
            m[1] += add
            new_index = choice(N[1],add,replace=False,p=z_1)
            new_score = np.sum(np.sum(np.abs(np.diff(A[:,new_index,:],axis=2)),axis=2)/N[2]/N[0],axis=0) + np.sum(np.sum(np.abs(np.diff(A[:,new_index,:],axis=0)),axis=0)/N[0]/N[2],axis=1)
            index_1 = np.append(index_1,new_index)
            score_1 = np.append(score_1,new_score)
            z_1 = linear(index_1,score_1,N[1])
            params[1] = entropy(score_1)
        
        if z_2 is not None:
            if np.sum(z_2>0) > theta[2]:
                add = theta[2]
            else:
                add = np.sum(z_2>0)
            m[2] += add
            new_index = choice(N[2],add,replace=False,p=z_2)
            new_score = np.sum(np.sum(np.abs(np.diff(A[:,:,new_index],axis=0)),axis=0)/N[0]/N[1],axis=0) + np.sum(np.sum(np.abs(np.diff(A[:,:,new_index],axis=1)),axis=1)/N[1]/N[0],axis=0)
            index_2 = np.append(index_2,new_index)
            score_2 = np.append(score_2,new_score)
            z_2 = linear(index_2, score_2, N[2])
            params[2] = entropy(score_2)
            
        index_0_log.append(index_0.tolist())
        index_1_log.append(index_1.tolist())
        index_2_log.append(index_2.tolist())
            
    return index_0_log, index_1_log, index_2_log
    
def rdskt_log(A,r,start,step,stop):
    N = np.array(A.shape)
    m = np.array([start,start,start],dtype=np.int)
    step = int((stop - start)/step)
    M = np.array(dirichlet([1,1,1],1)[0] * stop,dtype=np.int)
    # checking impossible sampling ratio
    while (M[0] > N[0]) or (M[1] > N[1]) or (M[2] > N[2]):
        M = np.array(dirichlet([1,1,1],1)[0] * stop,dtype=np.int)
    Theta = np.linspace(m,M,step)
    
    # inital core uniform sampling
    index_0 = np.append(choice(N[0],start-2,replace=False),[0,N[0]-1])
    index_1 = np.append(choice(N[1],start-2,replace=False),[0,N[1]-1])
    index_2 = np.append(choice(N[2],start-2,replace=False),[0,N[2]-1])
    index_0_log = [index_0.tolist()]
    index_1_log = [index_1.tolist()]
    index_2_log = [index_2.tolist()]
    
    # bilinear interpolation
    z_0 = np.arange(N[0])
    z_0[index_0] = 0
    z_0 = z_0/np.sum(z_0)
    
    z_1 = np.arange(N[1])
    z_1[index_1] = 0
    z_1 = z_1/np.sum(z_1)
    
    z_2 = np.arange(N[2])
    z_2[index_2] = 0
    z_2 = z_2/np.sum(z_2)
    
    remainder = np.array([0,0,0])
    for i in range(step-1):
        Theta[i] -= remainder
        Theta[i+1] += remainder
        diff = Theta[i+1] - Theta[i]
        remainder = diff - diff.astype(int)
        theta = diff.astype(int)
        
        if z_0 is not None:
            if np.sum(z_0>0) > theta[0]:
                add = theta[0]
            else:
                add = np.sum(z_0>0)
            m[0] += add
            index_0 = np.append(index_0,choice(N[0],add,replace=False,p=z_0))
            z_0[index_0] = 0
            if np.sum(z_0) > 0:
                z_0 = z_0/np.sum(z_0)
            if np.sum(z_0) == 0:
                z_0 = None
            
        if z_1 is not None:
            if np.sum(z_1>0) > theta[1]:
                add = theta[1]
            else:
                add = np.sum(z_1>0)
            m[1] += add
            index_1 = np.append(index_1,choice(N[1],add,replace=False,p=z_1))
            z_1[index_1] = 0
            if np.sum(z_1) > 0:
                z_1 = z_1/np.sum(z_1)
            if np.sum(z_1) == 0:
                z_1 = None

        if z_2 is not None:
            if np.sum(z_2>0) > theta[2]:
                add = theta[2]
            else:
                add = np.sum(z_2>0)
            m[2] += add
            index_2 = np.append(index_2,choice(N[2],add,replace=False,p=z_2))
            z_2[index_2] = 0
            if np.sum(z_2) > 0:
                z_2 = z_2/np.sum(z_2)
            if np.sum(z_2) == 0:
                z_2 = None
                
        index_0_log.append(index_0.tolist())
        index_1_log.append(index_1.tolist())
        index_2_log.append(index_2.tolist())
            
    return index_0_log, index_1_log, index_2_log
    
def lc(N, logPS_0, logPS_1, logPS_2, mode):
    err_log = []
    memory_log = []
    for index_0, index_1, index_2 in zip(logPS_0,logPS_1,logPS_2):
        m = [len(index_0),len(index_1),len(index_2)]
        
        if mode == 'progressive':
            y0, x0 = np.meshgrid(index_1,index_2)
            y1, x1 = np.meshgrid(index_2,index_0)
            y2, x2 = np.meshgrid(index_0,index_1)
            fibers_0 = (x0 + y0*N[2]).reshape(-1)
            fibers_1 = (x1 + y1*N[0]).reshape(-1)
            fibers_2 = (x2 + y2*N[1]).reshape(-1)
        elif mode == 'random':
            fibers_0 = choice(N[1]*N[2],m[1]*m[2],replace=False)
            fibers_1 = choice(N[2]*N[0],m[2]*m[0],replace=False)
            fibers_2 = choice(N[0]*N[1],m[0]*m[1],replace=False)

        k = np.array(r + (m-r)/3,dtype=np.int) 
        s = np.array(r + 2*(m-r)/3,dtype=np.int)

        Map0_T = normal(size=(len(fibers_0),k[0]))
        Map1_T = normal(size=(len(fibers_1),k[1]))
        Map2_T = normal(size=(len(fibers_2),k[2]))

        Y0 = A0[:,fibers_0] @ Map0_T
        Y1 = A1[:,fibers_1] @ Map1_T
        Y2 = A2[:,fibers_2] @ Map2_T

        Q0, _ = qr(np.array(Y0))
        Q1, _ = qr(np.array(Y1))
        Q2, _ = qr(np.array(Y2))

        Map0 = normal(size=(s[0],m[0]))
        Map1 = normal(size=(s[1],m[1]))
        Map2 = normal(size=(s[2],m[2]))

        delta0, delta1, delta2 = np.meshgrid(index_0, index_1, index_2, indexing='ij')
        Z = np.tensordot(np.tensordot(np.tensordot(A[delta0,delta1,delta2],Map0,axes=(0,1)),
                                      Map1,axes=(0,1)),
                         Map2,axes=(0,1))

        Q0, _ = qr(np.array(Y0))
        Q1, _ = qr(np.array(Y1))
        Q2, _ = qr(np.array(Y2))

        C = np.tensordot(np.tensordot(np.tensordot(Z,pinv(np.dot(Map0,Q0[index_0,:])),axes=(0,1)),
                                      pinv(np.dot(Map1,Q1[index_1,:])),axes=(0,1)),
                         pinv(np.dot(Map2,Q2[index_2,:])),axes=(0,1))

        S, U1, U2, U3 = HOSVD(C)

        Cr = np.tensordot(np.tensordot(np.tensordot(S[:r[0],:r[1],:r[2]],U1[:,:r[0]],axes=(0,1)),
                                       U2[:,:r[1]],axes=(0,1)),
                          U3[:,:r[2]],axes=(0,1))

        Ar = np.tensordot(np.tensordot(np.tensordot(Cr,Q0,axes=(0,1)),
                                       Q1,axes=(0,1)),
                          Q2,axes=(0,1))

        err = norm(A - Ar)**2 / norm(A)**2
        
        memory_log.append(len(fibers_0) * N[0] + len(fibers_1) * N[1] + len(fibers_2) * N[2] - 2*A[delta0,delta1,delta2].size)
        
        err_log.append(err)
        
    return err_log, memory_log
    
def Ahat(N, index_0, index_1, index_2):

    m = [len(index_0),len(index_1),len(index_2)]

    y0, x0 = np.meshgrid(index_1,index_2)
    y1, x1 = np.meshgrid(index_2,index_0)
    y2, x2 = np.meshgrid(index_0,index_1)
    fibers_0 = (x0 + y0*N[2]).reshape(-1)
    fibers_1 = (x1 + y1*N[0]).reshape(-1)
    fibers_2 = (x2 + y2*N[1]).reshape(-1)

    k = np.array(r + (m-r)/3,dtype=np.int) 
    s = np.array(r + 2*(m-r)/3,dtype=np.int)

    Map0_T = normal(size=(len(fibers_0),k[0]))
    Map1_T = normal(size=(len(fibers_1),k[1]))
    Map2_T = normal(size=(len(fibers_2),k[2]))

    Y0 = A0[:,fibers_0] @ Map0_T
    Y1 = A1[:,fibers_1] @ Map1_T
    Y2 = A2[:,fibers_2] @ Map2_T

    Q0, _ = qr(np.array(Y0))
    Q1, _ = qr(np.array(Y1))
    Q2, _ = qr(np.array(Y2))

    Map0 = normal(size=(s[0],m[0]))
    Map1 = normal(size=(s[1],m[1]))
    Map2 = normal(size=(s[2],m[2]))

    delta0, delta1, delta2 = np.meshgrid(index_0, index_1, index_2, indexing='ij')
    Z = np.tensordot(np.tensordot(np.tensordot(A[delta0,delta1,delta2],Map0,axes=(0,1)),
                                  Map1,axes=(0,1)),
                     Map2,axes=(0,1))

    Q0, _ = qr(np.array(Y0))
    Q1, _ = qr(np.array(Y1))
    Q2, _ = qr(np.array(Y2))

    C = np.tensordot(np.tensordot(np.tensordot(Z,pinv(np.dot(Map0,Q0[index_0,:])),axes=(0,1)),
                                  pinv(np.dot(Map1,Q1[index_1,:])),axes=(0,1)),
                     pinv(np.dot(Map2,Q2[index_2,:])),axes=(0,1))

    S, U1, U2, U3 = HOSVD(C)

    Cr = np.tensordot(np.tensordot(np.tensordot(S[:r[0],:r[1],:r[2]],U1[:,:r[0]],axes=(0,1)),
                                   U2[:,:r[1]],axes=(0,1)),
                      U3[:,:r[2]],axes=(0,1))

    Ar = np.tensordot(np.tensordot(np.tensordot(Cr,Q0,axes=(0,1)),
                                   Q1,axes=(0,1)),
                      Q2,axes=(0,1))
        
    return Ar

mat = scipy.io.loadmat('../data/Cardiac_DE_2.mat')
A = mat['X']
N = A.shape
r = np.array([20,20,5])

# matrix unfoldings
A0 = A.reshape(N[0],N[1]*N[2]) # 0 1 2
tmp = np.moveaxis(A,0,2) # 1 2 0
A1 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])
tmp = np.moveaxis(A,2,0) # 2 0 1
A2 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])

for it in tqdm(range(100)):
    np.random.seed(it)
    logPS_0, logPS_1, logPS_2 = pgskt_log(A,r,10,10,500)
    logRS_0, logRS_1, logRS_2 = rdskt_log(A,r,10,10,500)
    err_logPS, memory_logPS = lc(N,logPS_0, logPS_1, logPS_2, 'progressive')
    err_logRS, memory_logRS = lc(N,logRS_0, logRS_1, logRS_2, 'random')
    nf_PS = [len(i)+len(j)+len(k) for i,j,k in zip(logPS_0,logPS_1,logPS_2)]
    nf_RS = [len(i)+len(j)+len(k) for i,j,k in zip(logRS_0,logRS_1,logRS_2)]
    np.save(f'outputs_AISTATS/Cardiac_err_log_PSCT_{it}.npy',np.array(err_logPS))
    np.save(f'outputs_AISTATS/Cardiac_err_log_RSCT_{it}.npy',np.array(err_logRS))
    np.save(f'outputs_AISTATS/Cardiac_memory_log_PSCT_{it}.npy',np.array(memory_logPS))
    np.save(f'outputs_AISTATS/Cardiac_memory_log_RSCT_{it}.npy',np.array(memory_logRS))
    np.save(f'outputs_AISTATS/Cardiac_nf_PSCT_{it}.npy',np.array(nf_PS))
    np.save(f'outputs_AISTATS/Cardiac_nf_RSCT_{it}.npy',np.array(nf_RS))
