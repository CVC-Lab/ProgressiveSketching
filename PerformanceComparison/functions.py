import numpy as np
from numpy.random import normal, choice, dirichlet
from numpy.linalg import norm, svd, qr
from scipy.linalg import pinv
from scipy.linalg import qr as qr2
from scipy.linalg.interpolative import interp_decomp
from scipy import interpolate
from scipy.stats import entropy
import time


'''
full-scan algorithms
'''

def HOSVD_rank(A):
    A1 = A.reshape(A.shape[0],A.shape[1]*A.shape[2]) # 1 2 3
    tmp = np.moveaxis(A,0,2) # 2 3 1
    A2 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])
    tmp = np.moveaxis(A,2,0) # 3 1 2
    A3 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])
    U1, s1, V1 = svd(A1,full_matrices=False)
    U2, s2, V2 = svd(A2,full_matrices=False)
    U3, s3, V3 = svd(A3,full_matrices=False)
    
    scree1 = []
    scree2 = []
    scree3 = []
    for r in range(s1.shape[0]):
        print(r,end='\r')
        scree1.append(norm(s1[r+1:])**2 / norm(A1)**2)
        if norm(s1[r+1:])**2 / norm(A1)**2 < 1e-7 :
                break
    for r in range(s2.shape[0]):
        print(r,end='\r')
        scree2.append(norm(s2[r+1:])**2 / norm(A2)**2)
        if norm(s2[r+1:])**2 / norm(A2)**2 < 1e-7 :
                break
    for r in range(s3.shape[0]):
        print(r,end='\r')
        scree3.append(norm(s3[r+1:])**2 / norm(A3)**2)   
        if norm(s3[r+1:])**2 / norm(A3)**2 < 1e-7 :
                break
                
    return [scree1, scree2, scree3]

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

def RP_HOSVD(A, A1, A2, A3, r):  
    # computation time log
    start_time = time.time()
    
    Map1 = normal(size=(A.shape[1]*A.shape[2],r[0]))
    Map2 = normal(size=(A.shape[2]*A.shape[0],r[1]))
    Map3 = normal(size=(A.shape[0]*A.shape[1],r[2]))
    
    W1 = A1 @ Map1
    W2 = A2 @ Map2
    W3 = A3 @ Map3
    
    Q1, _ = qr(W1)
    Q2, _ = qr(W2)
    Q3, _ = qr(W3)
    
    S = np.tensordot(np.tensordot(np.tensordot(A,Q1.T,axes=(0,1)),
                          Q2.T,axes=(0,1)),
             Q3.T,axes=(0,1))
    
    cputime = time.time() - start_time  
    
    Ar = np.tensordot(np.tensordot(np.tensordot(S,Q1,axes=(0,1)),
                                   Q2,axes=(0,1)),
                      Q3,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return err, cputime
    
def RP_HOOI(A,r,tol=1e-5):
    # computation time log
    start_time = time.time()
    Q0_prev = np.zeros((A.shape[0],r[0]))
    
    Q0 = normal(size=(A.shape[0],r[0]))
    Q1 = normal(size=(A.shape[1],r[1]))
    Q2 = normal(size=(A.shape[2],r[2]))
    
    while norm(Q0_prev-Q0) > tol:
        
        Q0_prev = Q0.copy()
        
        #n=1
        S = np.tensordot(np.tensordot(A,Q1.T,axes=(1,1)),Q2.T,axes=(1,1))
        S0 = S.reshape(S.shape[0],S.shape[1]*S.shape[2]) # 1 2 3
        
        Map0 = normal(size=(r[1]*r[2],r[0]))
        W0 = S0 @ Map0
        Q0, _ = qr(W0)

        #n=2
        S = np.tensordot(np.tensordot(A,Q2.T,axes=(2,1)),Q0.T,axes=(0,1))
        S1 = S.reshape(S.shape[0],S.shape[1]*S.shape[2]) # 1 2 3
        
        Map1 = normal(size=(r[2]*r[0],r[1]))
        W1 = S1 @ Map1
        Q1, _ = qr(W1)

        #n=3
        S = np.tensordot(np.tensordot(A,Q0.T,axes=(0,1)),Q1.T,axes=(0,1))
        S2 = S.reshape(S.shape[0],S.shape[1]*S.shape[2]) # 1 2 3
        
        Map2 = normal(size=(r[0]*r[1],r[2]))
        W2 = S2 @ Map2
        Q2, _ = qr(W2)

        S = np.tensordot(np.tensordot(np.tensordot(A,Q0.T,axes=(0,1)),
                              Q1.T,axes=(0,1)),
                 Q2.T,axes=(0,1))
        
    cputime = time.time() - start_time 
    
    Ar = np.tensordot(np.tensordot(np.tensordot(S,Q0,axes=(0,1)),
                                   Q1,axes=(0,1)),
                      Q2,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return err, cputime

def RSVD(A,r,p=5,q=1):
    Map = normal(size=[A.shape[1],(p+r)])
    Y = np.linalg.matrix_power(A @ A.T,q) @ A @ Map
    Q,_ = qr(Y)
    B = Q.T @ A
    U, s, V = svd(B,full_matrices=False)
    Utilde = Q @ U
    return Utilde[:,:r], s[:r], V[:,:r]

def RP_STHOSVD(A,A1,A2,A3,r):
    # computation time log
    start_time = time.time()
    
    Q1, _, _ = RSVD(A1,r[0])
    S = np.tensordot(A,Q1.T,axes=(0,1))
    Q2, _, _ = RSVD(A2,r[1])
    S = np.tensordot(S,Q2.T,axes=(0,1))
    Q3, _, _ = RSVD(A3,r[2])
    S = np.tensordot(S,Q3.T,axes=(0,1))
    
    cputime = time.time() - start_time  
    
    Ar = np.tensordot(np.tensordot(np.tensordot(S,Q1,axes=(0,1)),
                                   Q2,axes=(0,1)),
                      Q3,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return err, cputime

def R_PET(A,A1,A2,A3,r):

    k = 2*r+1
    s = 2*k+1
    
    # computation time log
    start_time = time.time()
    
    Map1 = normal(size=(A.shape[1]*A.shape[2],k[0]))
    Map2 = normal(size=(A.shape[2]*A.shape[0],k[1]))
    Map3 = normal(size=(A.shape[0]*A.shape[1],k[2]))
    
    Y1 = A1 @ Map1
    Y2 = A2 @ Map2
    Y3 = A3 @ Map3
    
    Q1, _ = qr(Y1)
    Q2, _ = qr(Y2)
    Q3, _ = qr(Y3)
    
    Map1 = normal(size=(s[0],A.shape[0]))
    Map2 = normal(size=(s[1],A.shape[1]))
    Map3 = normal(size=(s[2],A.shape[2]))
    
    Z = np.tensordot(np.tensordot(np.tensordot(A,Map1,axes=(0,1)),
                                  Map2,axes=(0,1)),
                     Map3,axes=(0,1))
    
    C = np.tensordot(np.tensordot(np.tensordot(Z,pinv(np.dot(Map1,Q1)),axes=(0,1)),
                                  pinv(np.dot(Map2,Q2)),axes=(0,1)),
                     pinv(np.dot(Map3,Q3)),axes=(0,1))
    
    S, U1, U2, U3 = HOSVD(C)
    
    Cr = np.tensordot(np.tensordot(np.tensordot(S[:r[0],:r[1],:r[2]],U1[:,:r[0]],axes=(0,1)),
                                   U2[:,:r[1]],axes=(0,1)),
                      U3[:,:r[2]],axes=(0,1))
    
    cputime = time.time() - start_time  
    
    Ar = np.tensordot(np.tensordot(np.tensordot(S,Q1,axes=(0,1)),
                               Q2,axes=(0,1)),
                  Q3,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return err, cputime

def R_ST(A,A1,A2,A3,r):
    # computation time log
    start_time = time.time()
    
    Q1 = A1[:,choice(A.shape[1]*A.shape[2],r[0],replace=False)]
    Q2 = A2[:,choice(A.shape[2]*A.shape[0],r[1],replace=False)]
    Q3 = A3[:,choice(A.shape[0]*A.shape[1],r[2],replace=False)]
    
    S = np.tensordot(np.tensordot(np.tensordot(A,pinv(Q1),axes=(0,1)),
                                  pinv(Q2),axes=(0,1)),
                     pinv(Q3),axes=(0,1))
    
    cputime = time.time() - start_time  
    
    Ar = np.tensordot(np.tensordot(np.tensordot(S,Q1,axes=(0,1)),
                               Q2,axes=(0,1)),
                  Q3,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return err, cputime

def R_HOID(A,A1,A2,A3,r):

    k = 2*r+1
    
    # computation time log
    start_time = time.time()
    
    Map1 = normal(size=(k[0],A.shape[0]))
    Map2 = normal(size=(k[1],A.shape[1]))
    Map3 = normal(size=(k[2],A.shape[2]))
    
    Y1 = Map1 @ A1
    Y2 = Map2 @ A2
    Y3 = Map3 @ A3
    
    idx1, _ = interp_decomp(Y1,eps_or_k=r[0])
    idx2, _ = interp_decomp(Y2,eps_or_k=r[1])
    idx3, _ = interp_decomp(Y3,eps_or_k=r[2])
    
    Q1 = A1[:,idx1[:r[0]]]
    Q2 = A2[:,idx2[:r[1]]]
    Q3 = A3[:,idx3[:r[2]]]
        
    S = np.tensordot(np.tensordot(np.tensordot(A,pinv(Q1),axes=(0,1)),
                                  pinv(Q2),axes=(0,1)),
                     pinv(Q3),axes=(0,1))
    
    cputime = time.time() - start_time  
    
    Ar = np.tensordot(np.tensordot(np.tensordot(S,Q1,axes=(0,1)),
                               Q2,axes=(0,1)),
                  Q3,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return err, cputime
    
def R_LSHOOI(A,A0,A1,A2,r,tol=1e-3,K=10,I=1e4):
    
    J1 = 10 * np.max(r)**2
    J2 = 10 * np.max(r)**3
    
    # computation time log
    start_time = time.time()
    
    Q0 = uniform(low=-1,high=1,size=(A.shape[0],r[0]))
    Q1 = uniform(low=-1,high=1,size=(A.shape[1],r[1]))
    Q2 = uniform(low=-1,high=1,size=(A.shape[2],r[2]))
    S = uniform(low=-1,high=1,size=(r[0],r[1],r[2]))
    S_prev = np.zeros((A.shape[0],r[0]))
    
    hashedIndices = np.random.choice(I**2, J1, replace=True)
    randSigns = np.random.choice(2, J1, replace=True) * 2 - 1
    matrixA = matrixA * randSigns.reshape(1, n)
    for i in range(I):
        idx = (hashedIndices == i)
        matrixC[:, i] = np.sum(matrixA[:, idx], 1)
        
    while norm(Q0_prev-Q0) > tol:
        
        Q0_prev = Q0.copy()
        
        S0 = S.reshape(S.shape[0],S.shape[1]*S.shape[2]) # 1 2 3
        a0 = np.kron(Q2,Q1) @ S0.T
        Q0 = lstsq(a0,A0.T)
        
        tmp = np.moveaxis(S,0,2) # 2 3 1
        S1 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])
        a1 = np.kron(Q2,Q0) @ S1.T
        Q1 = lstsq(a1,A1.T)
        
        tmp = np.moveaxis(S,2,0) # 3 1 2
        S2 = tmp.reshape(tmp.shape[0],tmp.shape[1]*tmp.shape[2])
        a2 = np.kron(Q1,Q0) @ S2.T
        Q2 = lstsq(a2,A2.T)
        
        b = np.kron(np.kron(Q2,Q1),Q0)
        S = lstsq(b,A2.T)
        
    cputime = time.time() - start_time 
    
    Ar = np.tensordot(np.tensordot(np.tensordot(S,Q0,axes=(0,1)),
                                   Q1,axes=(0,1)),
                      Q2,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return err, cputime

def SCSVD(A,r,p,q,k=None,s=None):
    M = A.shape[0]
    N = A.shape[1]
    if k is None:
        k = 4*r + 1
    if s is None:
        s = 2*k + 1
        
    m = int(M*p)
    n = int(N*q)

    rows = np.sort(choice(M, size=m, replace=False))
    cols = np.sort(choice(N, size=n, replace=False))
    
    Gamma = normal(size=(k,m))
    Omega = normal(size=(k,n))
    Phi = normal(size=(s,m))
    Psi = normal(size=(s,n))

    X = Gamma @ A[rows,:]
    Y = A[:,cols] @ Omega.T
    Z = Phi @ A[np.ix_(rows,cols)] @ Psi.T

    P, _ = qr(np.array(X.T))
    Q, _ = qr(np.array(Y))

    C = pinv(Phi @ Q[rows,:]).dot(Z).dot(pinv(Psi @ P[cols,:]).T)
    U, s, VT = svd(C)
    Cr = U[:,:r].dot(np.diag(s[:r])).dot(VT[:r,:])
    return Q, Cr, P.T, rows, cols

def R_SCTT(A,rank,p):
    start_time = time.time()
    
    d = len(A.shape)
    r = [1]
    G = []
    C = A
    for k in range(1,d):
        C = C.reshape((int(r[k-1]*A.shape[k-1]),int(C.size/r[k-1]/A.shape[k-1])),order='F')
        U, s, VT, _, _ = SCSVD(C, rank[k-1], p, p)
        r.append(rank[k-1])
        U = U[:, :r[k]]
        s = s[:r[k],:r[k]]
        VT = VT[:r[k], :]
        G.append(U.reshape((r[k-1],A.shape[k-1],r[k]),order='F'))
        C = np.dot(s,VT)
    G.append(C[:,:,np.newaxis])
    
    cputime = time.time() - start_time 
    
    Ar = G[0].reshape(A.shape[0], -1)
    Ar = Ar.dot(G[1].reshape(r[1],-1)).reshape(-1,r[2])
    Ar = Ar.dot(G[2].reshape(r[2],-1)).reshape(A.shape[0],A.shape[1],A.shape[2])
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return err, cputime

'''
SketchyCoreTucker
'''
    
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
        
def pgSketchyCoreTucker(A,A0,A1,A2,r,start,step,stop):
    N = np.array(A.shape)
    m = np.array([start,start,start],dtype=int)
    params = [1,1,1]
    
    # computation time log
    start_time = time.time()
    
    # inital core uniform sampling
    index_0 = np.append(choice(N[0]-2,start-2,replace=False)+1,[0,N[0]-1])
    index_1 = np.append(choice(N[1]-2,start-2,replace=False)+1,[0,N[1]-1])
    index_2 = np.append(choice(N[2]-2,start-2,replace=False)+1,[0,N[2]-1])
    
    # order 2 SAD
    score_0 = np.sum(np.sum(np.abs(np.diff(A[index_0,:,:],axis=1)),axis=1)/N[1],axis=1) + np.sum(np.sum(np.abs(np.diff(A[index_0,:,:],axis=2)),axis=2)/N[2],axis=1)
    score_1 = np.sum(np.sum(np.abs(np.diff(A[:,index_1,:],axis=2)),axis=2)/N[2],axis=0) + np.sum(np.sum(np.abs(np.diff(A[:,index_1,:],axis=0)),axis=0)/N[0],axis=1)
    score_2 = np.sum(np.sum(np.abs(np.diff(A[:,:,index_2],axis=0)),axis=0)/N[0],axis=0) + np.sum(np.sum(np.abs(np.diff(A[:,:,index_2],axis=1)),axis=1)/N[1],axis=0)
    
    # bilinear interpolation
    z_0 = linear(index_0,score_0,N[0])
    z_1 = linear(index_1,score_1,N[1])
    z_2 = linear(index_2,score_2,N[2])
    
    while np.sum(m) < stop:
        theta = np.array(dirichlet(params,1)[0] * step,dtype=int)
        
        if z_0 is not None:
            if np.sum(z_0>0) > theta[0]:
                add = theta[0]
            else:
                add = np.sum(z_0>0)
            m[0] += add
            new_index = choice(N[0],add,replace=False,p=z_0)
            new_score = np.sum(np.sum(np.abs(np.diff(A[new_index,:,:],axis=1)),axis=1)/N[1],axis=1) + np.sum(np.sum(np.abs(np.diff(A[new_index,:,:],axis=2)),axis=2)/N[2],axis=1)
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
            new_score = np.sum(np.sum(np.abs(np.diff(A[:,new_index,:],axis=2)),axis=2)/N[2],axis=0) + np.sum(np.sum(np.abs(np.diff(A[:,new_index,:],axis=0)),axis=0)/N[0],axis=1)
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
            new_score = np.sum(np.sum(np.abs(np.diff(A[:,:,new_index],axis=0)),axis=0)/N[0],axis=0) + np.sum(np.sum(np.abs(np.diff(A[:,:,new_index],axis=1)),axis=1)/N[1],axis=0)
            index_2 = np.append(index_2,new_index)
            score_2 = np.append(score_2,new_score)
            z_2 = linear(index_2, score_2, N[2])
            params[2] = entropy(score_2)
            
    # Final SketchyCoreTucker
    y0, x0 = np.meshgrid(index_1,index_2)
    y1, x1 = np.meshgrid(index_2,index_0)
    y2, x2 = np.meshgrid(index_0,index_1)
    fibers_0 = (x0 + y0*N[2]).reshape(-1)
    fibers_1 = (x1 + y1*N[0]).reshape(-1)
    fibers_2 = (x2 + y2*N[1]).reshape(-1)
    
    k = np.array(r + (m-r)/3,dtype=int) 
    s = np.array(r + 2*(m-r)/3,dtype=int)
    
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
    
    C = np.tensordot(np.tensordot(np.tensordot(Z,pinv(np.dot(Map0,Q0[index_0,:])),axes=(0,1)),
                                  pinv(np.dot(Map1,Q1[index_1,:])),axes=(0,1)),
                     pinv(np.dot(Map2,Q2[index_2,:])),axes=(0,1))
    
    S, U1, U2, U3 = HOSVD(C)
    
    Cr = np.tensordot(np.tensordot(np.tensordot(S[:r[0],:r[1],:r[2]],U1[:,:r[0]],axes=(0,1)),
                                   U2[:,:r[1]],axes=(0,1)),
                      U3[:,:r[2]],axes=(0,1))
    
    time_TS = time.time() - start_time  
    
    Ar = np.tensordot(np.tensordot(np.tensordot(Cr,Q0,axes=(0,1)),
                                   Q1,axes=(0,1)),
                      Q2,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
    
    return Ar, err, m, time_TS
    
def Random_SketchyCoreTucker(A,A0,A1,A2,r,M):
    N = np.array(A.shape)
    m = np.array(dirichlet([1,1,1],1)[0] * M,dtype=int)
    # checking impossible sampling ratio
    while (m[0] > N[0]) or (m[1] > N[1]) or (m[2] > N[2]):
        m = np.array(dirichlet([1,1,1],1)[0] * M,dtype=int)
    k = np.array(r + (m-r)/3,dtype=int) 
    s = np.array(r + 2*(m-r)/3,dtype=int)
    
    # computation time log
    start_time = time.time()
    
    # inital core uniform sampling
    index_0 = choice(N[0], size=m[0], replace=False)
    index_1 = choice(N[1], size=m[1], replace=False) 
    index_2 = choice(N[2], size=m[2], replace=False)
    
    fibers_0 = choice(N[1]*N[2],m[1]*m[2],replace=False)
    fibers_1 = choice(N[2]*N[0],m[2]*m[0],replace=False)
    fibers_2 = choice(N[0]*N[1],m[0]*m[1],replace=False)
    
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
    
    C = np.tensordot(np.tensordot(np.tensordot(Z,pinv(np.dot(Map0,Q0[index_0,:])),axes=(0,1)),
                                  pinv(np.dot(Map1,Q1[index_1,:])),axes=(0,1)),
                     pinv(np.dot(Map2,Q2[index_2,:])),axes=(0,1))
    
    S, U1, U2, U3 = HOSVD(C)
    
    Cr = np.tensordot(np.tensordot(np.tensordot(S[:r[0],:r[1],:r[2]],U1[:,:r[0]],axes=(0,1)),
                                   U2[:,:r[1]],axes=(0,1)),
                      U3[:,:r[2]],axes=(0,1))
    
    time_TS = time.time() - start_time 
    
    Ar = np.tensordot(np.tensordot(np.tensordot(Cr,Q0,axes=(0,1)),
                                   Q1,axes=(0,1)),
                      Q2,axes=(0,1))
    
    err = norm(A - Ar)**2 / norm(A)**2
        #
    return Ar, err, time_TS
    
def pgskt_log(A,A0,A1,A2,r,start,step,stop):
    N = np.array(A.shape)
    m = np.array([start,start,start],dtype=int)
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
        theta = np.array(dirichlet(params,1)[0] * step,dtype=int)
        
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
    
def rdskt_log(A,A0,A1,A2,r,start,step,stop):
    N = np.array(A.shape)
    m = np.array([start,start,start],dtype=int)
    params = [1,1,1]
      
    # inital core uniform sampling
    index_0 = np.append(choice(N[0]-2,start-2,replace=False)+1,[0,N[0]-1])
    index_1 = np.append(choice(N[1]-2,start-2,replace=False)+1,[0,N[1]-1])
    index_2 = np.append(choice(N[2]-2,start-2,replace=False)+1,[0,N[2]-1])
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
        theta = np.array(dirichlet(params,1)[0] * step,dtype=int)

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
    
def lc(N,A,A0,A1,A2,r,logPS_0,logPS_1,logPS_2,mode):
    err_log = []
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

        k = np.array(r + (m-r)/3,dtype=int) 
        s = np.array(r + 2*(m-r)/3,dtype=int)

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
        
        err_log.append(err)
        
    return err_log
    
def Ahat(N,A,A0,A1,A2,r,index_0,index_1,index_2):

    m = [len(index_0),len(index_1),len(index_2)]

    y0, x0 = np.meshgrid(index_1,index_2)
    y1, x1 = np.meshgrid(index_2,index_0)
    y2, x2 = np.meshgrid(index_0,index_1)
    fibers_0 = (x0 + y0*N[2]).reshape(-1)
    fibers_1 = (x1 + y1*N[0]).reshape(-1)
    fibers_2 = (x2 + y2*N[1]).reshape(-1)

    k = np.array(r + (m-r)/3,dtype=int) 
    s = np.array(r + 2*(m-r)/3,dtype=int)

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
