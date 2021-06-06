#################################################################################

import numpy as np
from netCDF4 import Dataset
import scipy.io
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
def unfoldings(A):
    N = A.shape
    output = [A.reshape(N[0],np.int(A.size/N[0]))]
    for i in range(1,len(N)):
        A = np.moveaxis(A,0,-1)
        output.append(A.reshape(N[i],np.int(A.size/N[i])))
    return output
    
def HOSVD(A):
    [A0, A1, A2, A3] = unfoldings(A)
    U0, s0, _ = svd(A0,full_matrices=False)
    U1, s1, _ = svd(A1,full_matrices=False)
    U2, s2, _ = svd(A2,full_matrices=False)
    U3, s3, _ = svd(A3,full_matrices=False)
    
    S = np.tensordot(np.tensordot(np.tensordot(np.tensordot(A,U0.T,axes=(0,1)),
                                               U1.T,axes=(0,1)),
                                  U2.T,axes=(0,1)),
                     U3.T,axes=(0,1))
    return S, U0, U1, U2, U3
    
def linear(index,score,N):
    f = interpolate.interp1d(index, score)
    z = f(np.arange(N))
    z[index] = 0
    if np.sum(z) > 0:
        return z / np.sum(z)
    if np.sum(z) == 0:
        return None
    
def pgskt_log(A,r,start,step,stop):
    N = np.array(A.shape)
    m = np.array([start,start,start],dtype=np.int)
    params = [1,1,1]
    
    # matrix unfoldings
    #[A0, A1, A2, A3] = unfoldings(A)
    
    # inital core uniform sampling
    index_0 = np.append(choice(N[0],start-2,replace=False),[0,N[0]-1])
    index_1 = np.append(choice(N[1],start-2,replace=False),[0,N[1]-1])
    index_2 = np.append(choice(N[2],start-2,replace=False),[0,N[2]-1])
    index_0_log = [index_0.tolist()]
    index_1_log = [index_1.tolist()]
    index_2_log = [index_2.tolist()]
    
    # SAD
    score_0 = np.sum(np.abs(np.diff(A0[index_0,:],axis=1)),axis=1)
    score_1 = np.sum(np.abs(np.diff(A1[index_1,:],axis=1)),axis=1)
    score_2 = np.sum(np.abs(np.diff(A2[index_2,:],axis=1)),axis=1)
    
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
            index_0 = np.append(index_0,choice(N[0],add,replace=False,p=z_0))
            #Q0, _ = qr(A0[index_0,:])
            #score_0 = np.sum(Q0**2,axis=1)/np.sum(Q0**2)
            score_0 = np.sum(np.abs(np.diff(A0[index_0,:],axis=1)),axis=1)
            z_0 = linear(index_0,score_0,N[0])

            #reward = score_0[-add:] >= np.mean(score_0)
            #params[0] += np.sum(reward)
            params[0] = entropy(score_0)
            
        if z_1 is not None:
            if np.sum(z_1>0) > theta[1]:
                add = theta[1]
            else:
                add = np.sum(z_1>0)
            m[1] += add
            index_1 = np.append(index_1,choice(N[1],add,replace=False,p=z_1))
            #Q1, _ = qr(A1[index_1,:])
            #score_1 = np.sum(Q1**2,axis=1)/np.sum(Q1**2)
            score_1 = np.sum(np.abs(np.diff(A1[index_1,:],axis=1)),axis=1)
            z_1 = linear(index_1,score_1,N[1])

            #reward = score_1[-add:] >= np.mean(score_1)
            #params[1] += np.sum(reward)
            params[1] = entropy(score_1)
        
        if z_2 is not None:
            if np.sum(z_2>0) > theta[2]:
                add = theta[2]
            else:
                add = np.sum(z_2>0)
            m[2] += add
            index_2 = np.append(index_2,choice(N[2],add,replace=False,p=z_2))
            #Q2, _ = qr(A2[index_2,:])
            #score_2 = np.sum(Q2**2,axis=1)/np.sum(Q2**2)
            score_2 = np.sum(np.abs(np.diff(A2[index_2,:],axis=1)),axis=1)
            z_2 = linear(index_2, score_2, N[2])

            #reward = score_2[-add:] >= np.mean(score_2)
            #params[2] += np.sum(reward)
            params[2] = entropy(score_2)
            
        index_0_log.append(index_0.tolist())
        index_1_log.append(index_1.tolist())
        index_2_log.append(index_2.tolist())
            
    return index_0_log, index_1_log, index_2_log
    
def rdskt_log(A,r,start,step,stop):
    N = np.array(A.shape)
    m = np.array([start,start,start],dtype=np.int)
    params = [1,1,1]
    
    # matrix unfoldings
    #[A0, A1, A2, A3] = unfoldings(A)
    
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
    
    while np.sum(m) < stop:
        theta = np.array(dirichlet(params,1)[0] * step,dtype=np.int)

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
    
def lc(N, logPS_0, logPS_1, logPS_2):
    err_log = []
    for index_0, index_1, index_2 in zip(logPS_0,logPS_1,logPS_2):
        m = [len(index_0),len(index_1),len(index_2),2]
        index_3 = [0,1]
        
        z0, y0, x0 = np.meshgrid(index_1,index_2,index_3)
        z1, y1, x1 = np.meshgrid(index_2,index_3,index_0)
        z2, y2, x2 = np.meshgrid(index_3,index_0,index_1)
        z3, y3, x3 = np.meshgrid(index_0,index_1,index_2)
        
        fibers_0 = (x0 + y0*N[3] + z0*N[3]*N[2]).reshape(-1)
        fibers_1 = (x1 + y1*N[0] + z1*N[0]*N[3]).reshape(-1)
        fibers_2 = (x2 + y2*N[1] + z2*N[1]*N[0]).reshape(-1)
        fibers_3 = (x3 + y3*N[2] + z3*N[2]*N[1]).reshape(-1)

        k = np.array(r + (m-r)/3,dtype=np.int) 
        s = np.array(r + 2*(m-r)/3,dtype=np.int)

        Map0_T = normal(size=(len(fibers_0),k[0]))
        Map1_T = normal(size=(len(fibers_1),k[1]))
        Map2_T = normal(size=(len(fibers_2),k[2]))
        Map3_T = normal(size=(len(fibers_3),k[3]))

        Y0 = A0[:,fibers_0] @ Map0_T
        Y1 = A1[:,fibers_1] @ Map1_T
        Y2 = A2[:,fibers_2] @ Map2_T
        Y3 = A3[:,fibers_3] @ Map3_T

        Q0, _ = qr(np.array(Y0))
        Q1, _ = qr(np.array(Y1))
        Q2, _ = qr(np.array(Y2))
        Q3, _ = qr(np.array(Y3))

        Map0 = normal(size=(s[0],m[0]))
        Map1 = normal(size=(s[1],m[1]))
        Map2 = normal(size=(s[2],m[2]))
        Map3 = normal(size=(s[3],m[3]))

        delta0, delta1, delta2, delta3 = np.meshgrid(index_0, index_1, index_2, index_3, indexing='ij')
        Z = np.tensordot(np.tensordot(np.tensordot(np.tensordot(A[delta0,delta1,delta2,delta3],Map0,axes=(0,1)),
                                                   Map1,axes=(0,1)),
                                      Map2,axes=(0,1)),
                         Map3,axes=(0,1))

        C = np.tensordot(np.tensordot(np.tensordot(np.tensordot(Z,pinv(np.dot(Map0,Q0[index_0,:])),axes=(0,1)),
                                                   pinv(np.dot(Map1,Q1[index_1,:])),axes=(0,1)),
                                      pinv(np.dot(Map2,Q2[index_2,:])),axes=(0,1)),
                         pinv(np.dot(Map3,Q3[index_3,:])),axes=(0,1))

        S, U0, U1, U2, U3 = HOSVD(C)

        Cr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(S[:r[0],:r[1],:r[2]],U0[:,:r[0]],axes=(0,1)),
                                                    U1[:,:r[1]],axes=(0,1)),
                                       U2[:,:r[2]],axes=(0,1)),
                          U3[:,:r[3]],axes=(0,1))

        Ar = np.tensordot(np.tensordot(np.tensordot(np.tensordot(Cr,Q0,axes=(0,1)),
                                                    Q1,axes=(0,1)),
                                       Q2,axes=(0,1)),
                          Q3,axes=(0,1))

        err = norm(A - Ar)**2 / norm(A)**2
        
        err_log.append(err)
        
    return err_log
    
def Ahat(N, index_0, index_1, index_2, index_3 = [0,1]):

    m = [len(index_0),len(index_1),len(index_2),len(index_3)]

    z0, y0, x0 = np.meshgrid(index_1,index_2,index_3)
    z1, y1, x1 = np.meshgrid(index_2,index_3,index_0)
    z2, y2, x2 = np.meshgrid(index_3,index_0,index_1)
    z3, y3, x3 = np.meshgrid(index_0,index_1,index_2)

    fibers_0 = (x0 + y0*N[3] + z0*N[3]*N[2]).reshape(-1)
    fibers_1 = (x1 + y1*N[0] + z1*N[0]*N[3]).reshape(-1)
    fibers_2 = (x2 + y2*N[1] + z2*N[1]*N[0]).reshape(-1)
    fibers_3 = (x3 + y3*N[2] + z3*N[2]*N[1]).reshape(-1)

    k = np.array(r + (m-r)/3,dtype=np.int) 
    s = np.array(r + 2*(m-r)/3,dtype=np.int)

    Map0_T = normal(size=(len(fibers_0),k[0]))
    Map1_T = normal(size=(len(fibers_1),k[1]))
    Map2_T = normal(size=(len(fibers_2),k[2]))
    Map3_T = normal(size=(len(fibers_3),k[3]))

    Y0 = A0[:,fibers_0] @ Map0_T
    Y1 = A1[:,fibers_1] @ Map1_T
    Y2 = A2[:,fibers_2] @ Map2_T
    Y3 = A3[:,fibers_3] @ Map3_T

    Q0, _ = qr(np.array(Y0))
    Q1, _ = qr(np.array(Y1))
    Q2, _ = qr(np.array(Y2))
    Q3, _ = qr(np.array(Y3))

    Map0 = normal(size=(s[0],m[0]))
    Map1 = normal(size=(s[1],m[1]))
    Map2 = normal(size=(s[2],m[2]))
    Map3 = normal(size=(s[3],m[3]))

    delta0, delta1, delta2, delta3 = np.meshgrid(index_0, index_1, index_2, index_3, indexing='ij')
    Z = np.tensordot(np.tensordot(np.tensordot(np.tensordot(A[delta0,delta1,delta2,delta3],Map0,axes=(0,1)),
                                               Map1,axes=(0,1)),
                                  Map2,axes=(0,1)),
                     Map3,axes=(0,1))

    C = np.tensordot(np.tensordot(np.tensordot(np.tensordot(Z,pinv(np.dot(Map0,Q0[index_0,:])),axes=(0,1)),
                                               pinv(np.dot(Map1,Q1[index_1,:])),axes=(0,1)),
                                  pinv(np.dot(Map2,Q2[index_2,:])),axes=(0,1)),
                     pinv(np.dot(Map3,Q3[index_3,:])),axes=(0,1))

    S, U0, U1, U2, U3 = HOSVD(C)

    Cr = np.tensordot(np.tensordot(np.tensordot(np.tensordot(S[:r[0],:r[1],:r[2]],U0[:,:r[0]],axes=(0,1)),
                                                U1[:,:r[1]],axes=(0,1)),
                                   U2[:,:r[2]],axes=(0,1)),
                      U3[:,:r[3]],axes=(0,1))

    Ar = np.tensordot(np.tensordot(np.tensordot(np.tensordot(Cr,Q0,axes=(0,1)),
                                                Q1,axes=(0,1)),
                                   Q2,axes=(0,1)),
                      Q3,axes=(0,1))
        
    return Ar
    
ncfile = Dataset("uwnd.202010.nc")
uwnd = np.squeeze(ncfile.variables['uwnd'])
ncfile.close()
uwnd = uwnd[:,16,:252,1:337]

ncfile = Dataset("vwnd.202010.nc")
vwnd = np.squeeze(ncfile.variables['vwnd'])
ncfile.close()
vwnd = vwnd[:,16,:252,1:337]

wf = np.zeros((248,252,336,2))
wf[:,:,:,0] = uwnd
wf[:,:,:,1] = vwnd
A = wf
del wf, uwnd, vwnd
N = A.shape

r = np.array([150,40,40,2])

[A0, A1, A2, A3] = unfoldings(A)

for i in tqdm(range(10)):
    logPS_0, logPS_1, logPS_2 = pgskt_log(A,r,10,10,800)
    logRS_0, logRS_1, logRS_2 = rdskt_log(A,r,10,10,800)
    err_logPS = lc(N,logPS_0, logPS_1, logPS_2)
    err_logRS = lc(N,logRS_0, logRS_1, logRS_2)
    nf_PS = [len(i)+len(j)+len(k) for i,j,k in zip(logPS_0,logPS_1,logPS_2)]
    nf_RS = [len(i)+len(j)+len(k) for i,j,k in zip(logRS_0,logRS_1,logRS_2)]
    np.save(f'outputs/WindField_err_logPS_{i}.npy',np.array(err_logPS))
    np.save(f'outputs/WindField_err_logRS_{i}.npy',np.array(err_logRS))
    np.save(f'outputs/WindField_nf_PS_{i}.npy',np.array(nf_PS))
    np.save(f'outputs/WindField_nf_RS_{i}.npy',np.array(nf_RS))
