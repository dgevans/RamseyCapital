# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:05:36 2013

@author: dgevans
"""
import numpy as np
from utilities import interpolator_factory
from IPython.parallel import Reference
import itertools
from scipy.optimize import root
#from nag import root 
from scipy.spatial import cKDTree

P = np.array([[0.6,0.4],[0.4,0.6]])

beta = 0.95

g = np.array([0.17,0.17])

theta = np.array([[3.3,3.],[1.1,1.]])/4.2
thetai,theta1 = None,None

alpha = np.array([0.5,0.5])
alphai,alpha1 = None,None

pi = np.array([1.,1.])
pi1,pii = None,None

U,Uc,Ucc,Un,Unn = None,None,None,None,None
F,FK,FN,FKN,FKK = None,None,None,None,None

def calibrateFromParaStruct(Para):
    '''
    Calibrates from Para struct
    '''
    for var,value in Para.__dict__.items():
        globals()[var] = value
    setUtilityFunction(Para.Ufunc)
    setProductionFunction(Para.Ffunc)
    splitVariables()

def setUtilityFunction(Ufunc):
    '''
    Sets the utility function
    '''
    global U,Uc,Ucc,Un,Unn
    U,Uc,Ucc,Un,Unn = Ufunc.U,Ufunc.Uc,Ufunc.Ucc,Ufunc.Un,Ufunc.Unn

def setProductionFunction(Ffunc):
    '''
    Sets the production functions
    '''
    global F,FK,FN,FKN,FKK
    F,FK,FN,FKN,FKK = Ffunc.F,Ffunc.FK,Ffunc.FN,Ffunc.FKN,Ffunc.FKK
    
def splitVariables():
    '''
    Given the alpha,theta,pi split them
    '''
    I,S = len(alpha),len(P)
    global alpha1,alphai,theta1,thetai,pi1,pii
    theta1 = theta[0,:]
    thetai = theta[1:,:].reshape(I-1,-1)
    alpha1 = alpha[0]
    alphai = alpha[1:].reshape(I-1,-1)
    pi1 = pi[0]
    pii = pi[1:].reshape(I-1,-1)

splitVariables()

class iterate_on_policies(object):
    '''
    Class to iterate on the policy function
    '''
    def __init__(self,PF,mubar,rhobar):
        '''
        Inits with policy functions PF
        '''
        self.PF = PF
        self.mubar = mubar
        self.rhobar = rhobar
    
    def __call__(self,state,z0=None):
        '''
        Finds the policy rules that solves the first order conditions at a given
        state.
        '''
        I,S = len(theta),len(P)
        cstate,s_ = state
        mu_,rho_,K_ = cstate[:(I-1)].reshape(I-1,1),cstate[I-1:2*(I-1)].reshape(I-1,1),cstate[2*(I-1)]
        if z0==None:
            z0 = self.PF[s_](cstate)
        res = root(lambda z: self.policy_residuals_nobounds((mu_,rho_,K_,s_),z),z0[:len(z0)-I])
        if not res.success:
            return None
        else:
            c1,ci,n1,ni,Rk,x_,xk,rhotilde,K,mu,phi,lam,lam_k,nu,xi,eta = getQuantities(res.x)
            Uc1 = Uc(c1,n1)
            N = pi1*theta1*n1+np.sum(pii*thetai*ni,0)
            VK = P[s_,:].dot( xi*FK(K_,N) + nu*FKK(K_,N))
            Vrho = (lam*Uc1 + lam_k*Uc1*Rk).dot(P[s_,:])
            return np.hstack((
            c1,ci.flatten(),n1,ni.flatten(),Rk,x_.flatten(),xk.flatten(),
            rhotilde.flatten(),K,mu.flatten(),phi.flatten(),lam.flatten(),
            lam_k.flatten(),nu,xi,eta.flatten(),Vrho,VK            
            ))
            
    def test(self,state,z0=None):
        '''
        Finds the policy rules that solves the first order conditions at a given
        state.
        '''
        I,S = len(theta),len(P)
        cstate,s_ = state
        mu_,rho_,K_ = cstate[:(I-1)].reshape(I-1,1),cstate[I-1:2*(I-1)].reshape(I-1,1),cstate[2*(I-1)]
        if z0==None:
            z0 = self.PF[s_](cstate)
        res = root(lambda z: self.policy_residuals((mu_,rho_,K_,s_),z),z0[:len(z0)-I])
        if not res.success:
            return None
        else:
            c1,ci,n1,ni,Rk,x_,xk,rhotilde,K,mu,phi,lam,lam_k,nu,xi,eta = getQuantities(res.x)
            Uc1 = Uc(c1,n1)
            N = pi1*theta1*n1+np.sum(pii*thetai*ni,0)
            VK = P[s_,:].dot( xi*FK(K_,N) + nu*FKK(K_,N))
            Vrho = (lam*Uc1 + lam_k*Uc1*Rk).dot(P[s_,:])
            return np.hstack((
            c1,ci.flatten(),n1,ni.flatten(),Rk,x_.flatten(),xk.flatten(),
            rhotilde.flatten(),K,mu.flatten(),phi.flatten(),lam.flatten(),
            lam_k.flatten(),nu,xi,eta.flatten(),Vrho,VK            
            ))
            
    def policy_residuals(self,state,z):
        '''
        Computes the residuals for the FOC 
        '''
        I,S = len(theta),len(P)
        mu_,rho_,K_,s_ = state
        #unpack quantities and functions
        c1,ci,n1,ni,Rk,x_,xk,rhotilde,K,mu,phi,lam,lam_k,nu,xi,eta = getQuantities(z)
        mutilde = projectVariable(mu,self.mubar)
        rho = projectVariable(rhotilde,self.rhobar)
        barlam = np.vectorize(lambda r: max(r,0.))(rhotilde-rho)
        lambar = -np.vectorize(lambda r: min(r,0.))(rhotilde-rho)        
        
        c1f,cif,n1f,nif,Rkf,x_f,xkf,rhotildef,Kf,muf,phif,lamf,lam_kf,nuf,xif,etaf,Vrhof,VKf = getFunctions(self.PF)
        #compute marginal utilities
        Uc1,Uci = Uc(c1,n1),Uc(ci,ni)
        Ucc1,Ucci = Ucc(c1,n1),Ucc(ci,ni)
        Un1,Uni = Un(c1,n1),Un(ci,ni)
        Unn1,Unni = Unn(c1,n1),Unn(ci,ni)
        #Compute expectation terms
        EUci = np.dot(Uci,P[s_,:]).reshape(I-1,1)
        EUciRk = np.dot(Uci*Rk,P[s_,:]).reshape(I-1,1)
        EUc1 = np.dot(Uc1,P[s_,:])
        EUc1Rk = np.dot(Uc1*Rk,P[s_,:])
        N = pi1*theta1*n1+np.sum(pii*thetai*ni,0)
        C = pi1*c1+np.sum(pii*ci,0)
        #compute future variables
        x = np.zeros((I-1,S))
        Vrho_prime = np.zeros((I-1,S))
        VK_prime = np.zeros(S)
        for s in range(S):
            stateprime = np.hstack((mutilde[:,s],rho[:,s],K[s]))
            x[:,s] = x_f[s](stateprime)
            #Uc1prime = Uc(c1f[s](stateprime),n1f[s](stateprime))
            #Nprime = pi1*theta1*n1f[s](stateprime) + sum(pii*thetai*nif[s](stateprime),0)
            #VK_prime[s] = P[s,:].dot( xif[s](stateprime)*FK(K[s],Nprime) + nuf[s](stateprime)*FKK(K[s],Nprime))
            #Vrho_prime[:,s] = (lamf[s](stateprime)*Uc1prime +
            #   lam_kf[s](stateprime)*Uc1prime*Rkf[s](stateprime)).dot(P[s_,:])
            VK_prime[s] = VKf[s](stateprime)
            Vrho_prime[:,s] = Vrhof[s](stateprime).flatten()
        #compute residuals
        #first constraints
        res = np.zeros(z.shape)
        res[:(I-1)*S] = ((x_-xk)*Uci/(beta*EUci)+xk*Uci*Rk/(beta*EUciRk) - (Uci*ci+Uni*ni)
                        +rho*(Uc1*c1+Un1*n1) - x).flatten()
        iz = (I-1)*S
        res[iz:iz+(I-1)*S] = (Un1*thetai*rho - Uni*theta1).flatten()
        iz += (I-1)*S
        res[iz:iz+(I-1)] = (rho_*EUc1 - (Uc1*rho).dot(P[s_,:]).reshape(I-1,1)).flatten()
        iz += (I-1)
        res[iz:iz+(I-1)] = (rho_*EUc1Rk -(Uc1*rho*Rk).dot(P[s_,:]).reshape(I-1,1)).flatten()
        iz += (I-1)
        res[iz:iz+S] = FK(K_,N)-Rk
        iz += S
        res[iz:iz+S] = F(K_,N) - g - C - K
        iz +=  S
        res[iz:iz+(I-1)*S] = (Uc1*rho-Uci).flatten()
        iz += (I-1)*S
        #now FOC
        res[iz:iz+(I-1)*S] = (pii*alphai*Uci - mu*( Uci+ci*Ucci ) + (x_-xk)*Ucci/(beta*EUci) * (mu-mu_)
            +xk*Ucci*Rk/(beta*EUciRk)*(mu-mu_) - pii*xi-eta*Ucci).flatten()
        iz += (I-1)*S
        res[iz:iz+S] = pi1*alpha1*Uc1 + np.sum(rho*mu,0)*( Uc1+c1*Ucc1 ) + np.sum(lam*(rho_-rho)*Ucc1,0)\
            +np.sum(lam_k*(rho_-rho)*Rk*Ucc1,0)-pi1*xi+np.sum(rho*eta*Ucc1,0)
        iz += S
        res[iz:iz+(I-1)*S] = ( pii*alphai*Uni - mu*( Uni+ni*Unni ) - phi*theta1*Unni
            + nu*FKN(K_,N)*pii*thetai + xi*FN(K_,N)*pii*thetai).flatten()
        iz += (I-1)*S
        res[iz:iz+S] = pi1*alpha1*Un1 + np.sum(rho*mu,0)*( Un1+n1*Unn1 ) +np.sum(phi*thetai*rho,0)*Unn1\
            + nu*FKN(K_,N)*pi1*theta1 + xi*FN(K_,N)*pi1*theta1
        iz += S
        res[iz:iz+(I-1)] = (mu*Rk*Uci/EUciRk).dot(P[s_,:]) - (mu*Uci/EUci).dot(P[s_,:])
        iz += (I-1)
        
        res[iz:iz+S] = np.sum( (xk*Uci/EUciRk) * (mu-mu_) ,0) - nu
        iz += S
        res[iz:iz+S] = beta*VK_prime-xi
        iz += S
        res[iz:iz+(I-1)] = (mu*Uci/EUci).dot(P[s_,:])-mu_.flatten()
        iz += (I-1)
        res[iz:iz+(I-1)*S] = (beta*Vrho_prime - lam*Uc1 - lam_k*Uc1*Rk + eta*Uc1 +phi*Un1*thetai-barlam+lambar).flatten()
        return res
    def policy_residuals_nobounds(self,state,z):
        '''
        Computes the residuals for the FOC 
        '''
        I,S = len(theta),len(P)
        mu_,rho_,K_,s_ = state
        #unpack quantities and functions
        c1,ci,n1,ni,Rk,x_,xk,rho,K,mu,phi,lam,lam_k,nu,xi,eta = getQuantities(z)
        mutilde = mu #projectVariable(mu,self.mubar)
        #rho = projectVariable(rhotilde,self.rhobar)
        #barlam = np.vectorize(lambda r: max(r,0.))(rhotilde-rho)
        #lambar = -np.vectorize(lambda r: min(r,0.))(rhotilde-rho)        
        
        c1f,cif,n1f,nif,Rkf,x_f,xkf,rhotildef,Kf,muf,phif,lamf,lam_kf,nuf,xif,etaf,Vrhof,VKf = getFunctions(self.PF)
        #compute marginal utilities
        Uc1,Uci = Uc(c1,n1),Uc(ci,ni)
        Ucc1,Ucci = Ucc(c1,n1),Ucc(ci,ni)
        Un1,Uni = Un(c1,n1),Un(ci,ni)
        Unn1,Unni = Unn(c1,n1),Unn(ci,ni)
        #Compute expectation terms
        EUci = np.dot(Uci,P[s_,:]).reshape(I-1,1)
        EUciRk = np.dot(Uci*Rk,P[s_,:]).reshape(I-1,1)
        EUc1 = np.dot(Uc1,P[s_,:])
        EUc1Rk = np.dot(Uc1*Rk,P[s_,:])
        N = pi1*theta1*n1+np.sum(pii*thetai*ni,0)
        C = pi1*c1+np.sum(pii*ci,0)
        #compute future variables
        x = np.zeros((I-1,S))
        Vrho_prime = np.zeros((I-1,S))
        VK_prime = np.zeros(S)
        #VK_prime_test = np.zeros(S)
        for s in range(S):
            stateprime = np.hstack((mutilde[:,s],rho[:,s],K[s]))
            x[:,s] = x_f[s](stateprime)
            #Uc1prime = Uc(c1f[s](stateprime),n1f[s](stateprime))
            #Nprime = pi1*theta1*n1f[s](stateprime) + sum(pii*thetai*nif[s](stateprime),0)
            #VK_prime[s] = P[s,:].dot( xif[s](stateprime)*FK(K[s],Nprime) + nuf[s](stateprime)*FKK(K[s],Nprime))
            #Vrho_prime[:,s] = (lamf[s](stateprime)*Uc1prime +
            #   lam_kf[s](stateprime)*Uc1prime*Rkf[s](stateprime)).dot(P[s_,:])
            VK_prime[s] = VKf[s](stateprime)
            Vrho_prime[:,s] = Vrhof[s](stateprime).flatten()
            #pdb.set_trace()
        #compute residuals
        #first constraints
        res = np.zeros(z.shape)
        res[:(I-1)*S] = ((x_-xk)*Uci/(beta*EUci)+xk*Uci*Rk/(beta*EUciRk) - (Uci*ci+Uni*ni)
                        +rho*(Uc1*c1+Un1*n1) - x).flatten()
        iz = (I-1)*S
        res[iz:iz+(I-1)*S] = (Un1*thetai*rho - Uni*theta1).flatten()
        iz += (I-1)*S
        res[iz:iz+(I-1)] = (rho_*EUc1 - (Uc1*rho).dot(P[s_,:]).reshape(I-1,1)).flatten()
        iz += (I-1)
        res[iz:iz+(I-1)] = (rho_*EUc1Rk -(Uc1*rho*Rk).dot(P[s_,:]).reshape(I-1,1)).flatten()
        iz += (I-1)
        res[iz:iz+S] = FK(K_,N)-Rk
        iz += S
        res[iz:iz+S] = F(K_,N) - g - C - K
        iz +=  S
        res[iz:iz+(I-1)*S] = (Uc1*rho-Uci).flatten()
        iz += (I-1)*S
        #now FOC
        res[iz:iz+(I-1)*S] = (pii*alphai*Uci - mu*( Uci+ci*Ucci ) + (x_-xk)*Ucci/(beta*EUci) * (mu-mu_)
            +xk*Ucci*Rk/(beta*EUciRk)*(mu-mu_) - pii*xi-eta*Ucci).flatten()
        iz += (I-1)*S
        res[iz:iz+S] = pi1*alpha1*Uc1 + np.sum(rho*mu,0)*( Uc1+c1*Ucc1 ) + np.sum(lam*(rho_-rho)*Ucc1,0)\
            +np.sum(lam_k*(rho_-rho)*Rk*Ucc1,0)-pi1*xi+np.sum(rho*eta*Ucc1,0)
        iz += S
        res[iz:iz+(I-1)*S] = ( pii*alphai*Uni - mu*( Uni+ni*Unni ) - phi*theta1*Unni
            + nu*FKN(K_,N)*pii*thetai + xi*FN(K_,N)*pii*thetai).flatten()
        iz += (I-1)*S
        res[iz:iz+S] = pi1*alpha1*Un1 + np.sum(rho*mu,0)*( Un1+n1*Unn1 ) +np.sum(phi*thetai*rho,0)*Unn1\
            + nu*FKN(K_,N)*pi1*theta1 + xi*FN(K_,N)*pi1*theta1
        iz += S
        res[iz:iz+(I-1)] = (mu*Rk*Uci/EUciRk).dot(P[s_,:]) - (mu*Uci/EUci).dot(P[s_,:])
        iz += (I-1)
        res[iz:iz+S] = np.sum( (xk*Uci/EUciRk) * (mu-mu_) ,0) - nu
        iz += S
        #pdb.set_trace()
        res[iz:iz+S] = beta*VK_prime-xi
        iz += S
        res[iz:iz+(I-1)] = (mu*Uci/EUci).dot(P[s_,:])-mu_.flatten()
        iz += (I-1)
        res[iz:iz+(I-1)*S] = (beta*Vrho_prime - lam*Uc1 - lam_k*Uc1*Rk + eta*Uc1 +phi*Un1*thetai).flatten()
        return res
        
        
def getQuantities(z):
    '''
    From z get the quantities
    '''
    S = len(P)
    I = len(alpha)
    c1 = z[:S]
    ci = z[S:I*S].reshape((-1,S))
    n1 = z[I*S:I*S+S]
    ni = z[I*S+S:2*I*S].reshape((-1,S))
    iz = 2*I*S
    Rk = z[iz:iz+S]
    iz += S
    x_ = z[iz:iz+(I-1)].reshape((-1,1))
    iz += (I-1)
    xk = z[iz:iz+(I-1)].reshape(I-1,1)
    iz += (I-1)
    rhotilde = z[iz:iz+(I-1)*S].reshape((-1,S))
    iz += (I-1)*S
    K = z[iz:iz+S]
    iz += S
    mu = z[iz:iz+(I-1)*S].reshape((-1,S))
    iz += (I-1)*S
    phi = z[iz:iz+(I-1)*S].reshape((-1,S))
    iz += (I-1)*S
    lam = z[iz:iz+(I-1)].reshape((-1,1))
    iz += I-1
    lam_k = z[iz:iz+(I-1)].reshape((-1,1))
    iz += I-1
    nu = z[iz:iz+S]
    iz += S
    xi = z[iz:iz+S]
    iz += S
    eta = z[iz:iz+(I-1)*S].reshape((-1,S))
    iz += (I-1)*S
    if iz == len(z):
        return c1,ci,n1,ni,Rk,x_,xk,rhotilde,K,mu,phi,lam,lam_k,nu,xi,eta
    Vrho = z[iz:iz+I-1].reshape((I-1),1)
    iz += I-1
    VK = z[iz]
    return c1,ci,n1,ni,Rk,x_,xk,rhotilde,K,mu,phi,lam,lam_k,nu,xi,eta,Vrho,VK
    
def getXDV(PR):
    '''
    From z get the quantities
    '''
    z = PR.T
    S = len(P)
    I = len(alpha)
    c1 = z[:S]
    ci = z[S:I*S]
    n1 = z[I*S:I*S+S]
    ni = z[I*S+S:2*I*S]
    iz = 2*I*S
    Rk = z[iz:iz+S]
    iz += S
    x_ = z[iz:iz+(I-1)]
    iz += (I-1)
    xk = z[iz:iz+(I-1)]
    iz += (I-1)
    rhotilde = z[iz:iz+(I-1)*S]
    iz += (I-1)*S
    K = z[iz:iz+S]
    iz += S
    mu = z[iz:iz+(I-1)*S]
    iz += (I-1)*S
    phi = z[iz:iz+(I-1)*S]
    iz += (I-1)*S
    lam = z[iz:iz+(I-1)]
    iz += I-1
    lam_k = z[iz:iz+(I-1)]
    iz += I-1
    nu = z[iz:iz+S]
    iz += S
    xi = z[iz:iz+S]
    iz += S
    eta = z[iz:iz+(I-1)*S]
    iz += (I-1)*S
    Vrho = z[iz:iz+I-1]
    iz += I-1
    VK = z[iz]
    return np.vstack((x_,Vrho,VK))

def getFunctions(PF):
    '''
    Gets the functions from PF
    '''
    return itertools.izip(*[getQuantities(pf) for pf in PF])

def projectVariable(x,xbar):
    '''
    Projects a variable x onto the range [xbar[0],xbar[1]]
    '''
    fproj = np.vectorize(lambda x: max(min(x,xbar[1]),xbar[0]))
    return fproj(x)
    
def solveFailedRoot(PRf,Xsolved,PRsolved,state,n=5):
    '''
    Finds a root of state given alternative policy rules
    '''
    cstate,s_ = state
    ixs = cKDTree(Xsolved).query(cstate,k=4)[1]#find nearest neighbor
    for ix in ixs:
        z0 = PRsolved[ix]
        for alpha in np.linspace(0,1,5)[1:]:
            new_state = ((1-alpha)*Xsolved[ix]+alpha*cstate,s_)
            z0 = PRf(new_state,z0)
            if z0 == None:
                z0 = PRf(new_state)
                if z0 == None:
                    return None
        if z0 != None:
            return z0
    return None
    
def findFailedRoots(PRnew,PR,X,s_):
    '''
    Finds
    '''
    unsolved = filter(lambda x: x[1][0] == None,enumerate(itertools.izip(PR,X)))
    solved = filter(lambda x: x[1][0] != None,enumerate(itertools.izip(PR,X)))
    iSolved,PRXsolved = zip(*solved)
    PRsolved,Xsolved = map(np.vstack,zip(*PRXsolved))
    iUnsolved,PRXunsolved = zip(*unsolved)
    for iu,(pr,x) in unsolved:
        PR[iu] = solveFailedRoot(PRnew,Xsolved,PRsolved,(x,s_),n=50)
        print iu,PR[iu]
    #unsolved = filter(lambda x: x[1][0] == None,enumerate(itertools.izip(PR,X)))
    #num_unsolved = len(unsolved)
        
    
def solvePlannersProblem_parallel(PF,Para,X,mubar,rhobar):
    '''
    Solves the planners problem using IPython parallel.
    '''
    #sets up some parrallel stuff
    from IPython.parallel import Client
    from IPython.parallel import Reference
    c = Client()
    v_lb = c.load_balanced_view()
    v = c[:]
    v.block = True
    
    #calibrate model
    calibrateFromParaStruct(Para)
    v.apply(calibrateFromParaStruct,Para)
    interpolate3d = interpolator_factory(['spline']*3,[10,10,10],[3]*3)
    S = len(P)    
    
    #get old policy rules
    PRs_old = []
    for s in range(S):
        PRs_old.append(PF[s](X).T)
    
    #Do policy iteration
    diff = 1.
    while diff > 1e-5:
        v['PRnew'] = iterate_on_policies(PF,mubar,rhobar)
        PRnew = Reference('PRnew') #create a reference to the remote object
        
        PRs = []
        for s_ in range(S):
            domain = itertools.izip(X,[s_]*len(X))
            policies = v_lb.imap(PRnew,domain)
            temp = []
            for i,pr in enumerate(policies):
                temp.append(pr)
                print i
            PRs.append(np.vstack(temp))
            diff = max(diff,np.amax(np.abs(PRs[s_]-PRs_old[s_])))
        print diff
        PRf = []
        for s_ in range(S):
            PRf.append(interpolate3d(X,PRs[s_]))
            PRs_old[s_] = PRs[s_]
    return PRf
    
def solvePlannersProblemIID_parallel(PF,Para,X,mubar,rhobar,c):
    '''
    Solves the planners problem using IPython parallel.
    '''
    #sets up some parrallel stuff
    v_lb = c.load_balanced_view()
    v = c[:]
    v.block = True
    
    #calibrate model
    calibrateFromParaStruct(Para)
    v.apply(calibrateFromParaStruct,Para)
    interpolate3d = interpolator_factory(['spline']*3,[10,10,10],[3]*3)
    S = len(P)    
    
    #get old policy rules
    PRs_old = []
    for s in range(S):
        PRs_old.append(PF[s](X).T)
        
    #Do policy iteration
    diff = 1.
    while diff > 1e-5:
        PRnew = iterate_on_policies(PF,mubar,rhobar)
        v['PRnew'] = PRnew
        rPRnew = Reference('PRnew') #create a reference to the remote object
        
        PRs = []
        domain = itertools.izip(X,[0]*len(X))
        i_policies = v_lb.imap(rPRnew,domain)
        policies = []
        for i,pol in enumerate(i_policies):
            policies.append(pol)
        try:
            PRs = [np.vstack(policies)]*S
        except:
            findFailedRoots(PRnew,policies,X,0)
            PRs = [np.vstack(policies)]*S
        diff = np.amax(np.abs((PRs[0]-PRs_old[0])))
        print "diff: ",diff
        PF = []
        for s_ in range(S):
            PF.append(interpolate3d(X,PRs[s_]))
            PRs_old[s_] = PRs[s_]
    return PF
    
def iteratePlannersProblemIID_parallel(PF,Para,X,mubar,rhobar,c):
    '''
    Iterates on the planners problem
    '''
    #sets up some parrallel stuff
    v = c[:]
    v.block = True
    
    #calibrate model
    calibrateFromParaStruct(Para)
    v.apply(calibrateFromParaStruct,Para)
    S = len(P) 
    
    #perform iteration
    PRnew = iterate_on_policies(PF,mubar,rhobar)
    v['PRnew'] = PRnew
    rPRnew = Reference('PRnew') #create a reference to the remote object
    
    domain = itertools.izip(X,[0]*len(X))
    i_policies = v.imap(rPRnew,domain)
    policies = []
    for i,pol in enumerate(i_policies):
        policies.append(pol)
    return policies
    
def solvePlannersProblemIID(PF,Para,X,mubar,rhobar):
    '''
    Solves the planners problem using IPython parallel.
    '''
    
    #calibrate model
    calibrateFromParaStruct(Para)
    interpolate3d = interpolator_factory(['spline']*3,[10,10,10],[3]*3)
    S = len(P)    
    
    #get old policy rules
    PRs_old = []
    for s in range(S):
        PRs_old.append(PF[s](X).T)
        
    #Do policy iteration
    diff = 1.
    while diff > 1e-5:
        PRnew = iterate_on_policies(PF,mubar,rhobar)
        
        PRs = []
        domain = itertools.izip(X,[0]*len(X))
        i_policies = itertools.imap(PRnew,domain)
        policies = []
        for i,pol in enumerate(i_policies):
            policies.append(pol)
            if i%20 == 0:
                print i
        try:
            PRs = [np.vstack(policies)]*S
        except:
            findFailedRoots(PRnew,policies,X,0)
            PRs = [np.vstack(policies)]*S
        diff = np.amax(np.abs((PRs[0]-PRs_old[0])/PRs[0]))
        print "diff: ",diff
        PF = []
        for s_ in range(S):
            PF.append(interpolate3d(X,PRs[s_]))
            PRs_old[s_] = PRs[s_]
    return PF,PRs
        

    
def simulate(T,mu0,rho0,K0,s0,PF,mubar):
    '''
    Simulates the economy.
    '''
    S = len(P)
    PRHist = []
    taukHist = []
    sHist = [s0]
    cstate = np.hstack((mu0.flatten(),rho0.flatten(),K0))
    PRHist.append(PF[s0](cstate))
    for t in range(1,T+1):
        c1,ci,n1,ni,Rk,x_,xk,rhotilde,K,mu,phi,lam,lam_k,nu,xi,eta,Vrho,VK = \
            getQuantities(PRHist[-1])
        s_ = sHist[-1]
        s = np.random.choice(range(S),p=P[s_,:])
        mutilde = projectVariable(mu,mubar)
        cstate = np.hstack((mutilde[:,s],rhotilde[:,s],K[s]))
        sHist.append(s)
        PRHist.append(PF[s](cstate))
        
        #compute capital tax
        c1prime,_,n1prime,_,Rkprime,_,_,_,_,_,_,_,_,_,_,_,_,_ = getQuantities(PRHist[-1])
        Uc1,Uc1prime = Uc(c1,n1),Uc(c1prime,n1prime)
        EUc1Rkprime = P[s,:].dot(Uc1prime*Rkprime)
        taukHist.append( 1- Uc1[s]/EUc1Rkprime )
    return sHist[1:],PRHist[:-1],taukHist[:-1]
        
        
    
    