# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:52:00 2013

@author: dgevans
"""
import numpy as np
from utilities import interpolator_factory
from utilities import makeGrid_generic
from scipy.optimize import root

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
    N,S = len(alpha),len(P)
    global alpha1,alphai,theta1,thetai,pi1,pii
    theta1 = theta[0,:]
    thetai = theta[1:,:].reshape(N-1,-1)
    alpha1 = alpha[0]
    alphai = alpha[1:].reshape(N-1,-1)
    pi1 = pi[0]
    pii = pi[1:].reshape(N-1,-1)
    
splitVariables()

z0_guess = None

class iterate_on_policies(object):
    '''
    Iterates on policies to solve the complete markets problem.
    '''
    def __init__(self,mu,rho,PF,Kbar):
        '''
        Initializes the code to iterate on policies. Solves over K for a given
        mu and rho
        '''
        self.mu = mu
        self.rho = rho
        self.PF = PF
        self.Kbar = Kbar
        
    def __call__(self,K_):
        return self.findPolicies(np.atleast_1d(K_))
    
    def findPolicies(self,K_):
        '''
        Finds the solution to the policy equation
        '''
        global z0_guess
        I,S = len(theta),len(P)
        if not self.PF == None:
            z0 = self.PF(K_)
            res = root(lambda z: self.policy_residuals(K_,z),z0)
            if not res.success:
                if not z0_guess == None and len(z0) == len(z0_guess):
                    res = root(lambda z: self.policy_residuals(K_,z),z0_guess)
                    if not res.success:
                        return None
                else:
                    return None
            z0_guess = res.x
            return res.x
        else:
            z0 = np.hstack((np.ones(2*I*S),np.zeros(2*(I-1)*S+S)))
            res = root(lambda z: self.policy_residuals_end(K_,z),z0)
            if not res.success:
                if not z0_guess == None:
                    res = root(lambda z: self.policy_residuals_end(K_,z),z0_guess)
                    if not res.success:
                        return None
                else:
                    return None
            z0_guess = res.x
            c1,ci,n1,ni,phi,xi,eta = getQuantities_end(res.x)
            return np.hstack((c1,ci.flatten(),n1,ni.flatten(),K_*np.ones(S),phi.flatten(),xi.flatten(),eta.flatten()))
        
    def policy_residuals(self,K_,z):
        '''
        Computes the residuals of a policy choice z
        '''
        import pdb
        mu,rho = self.mu,self.rho
        I,S = len(theta),len(P)
        #unpack policies and functions
        c1,ci,n1,ni,Ktilde,phi,xi,eta = getQuantities(z)
        K = projectVariable(Ktilde,self.Kbar)
        barlam = np.vectorize(lambda k: max(k,0.))(Ktilde-K)
        lambar = -np.vectorize(lambda k: min(k,0.))(Ktilde-K) 
        
        
        c1f,cif,n1f,nif,Kf,phif,xif,etaf = getQuantities(self.PF)
        #compute marginal utilities
        N = pi1*theta1*n1+np.sum(pii*thetai*ni,0)
        Uc1,Uci = Uc(c1,n1),Uc(ci,ni)
        Ucc1,Ucci = Ucc(c1,n1),Ucc(ci,ni)
        Un1,Uni = Un(c1,n1),Un(ci,ni)
        Unn1,Unni = Unn(c1,n1),Unn(ci,ni)
        #compute future variable
        ExiFK = np.zeros(S)
        for s in range(S):
            Ks = np.array([K[s]])
            
            Nprime = pi1*theta1*n1f(Ks)+np.sum(pii*thetai*nif(Ks),0)
            ExiFK[s] = P[s,:].dot(xif(Ks)*FK(Ks,Nprime))
            #pdb.set_trace()
        #compute residuals
        res = np.zeros(len(z))
        res[:(I-1)*S] = (Un1*thetai*rho-Uni*theta1).flatten()
        iz = (I-1)*S
        res[iz:iz+S] = F(K_,N)-K-g-pi1*c1-np.sum(pii*ci,0)
        iz += S
        res[iz:iz+(I-1)*S] = (rho*Uc1 - Uci).flatten()
        iz += (I-1)*S
        res[iz:iz+(I-1)*S] = (alphai*pii*Uci - mu*(Uci+ci*Ucci) - pii*xi - eta*Ucci).flatten()
        iz += (I-1)*S
        res[iz:iz+S] = alpha1*pi1*Uc1 + np.sum(rho*mu,0)*( Uc1+c1*Ucc1 )-pi1*xi + np.sum(rho*eta,0)*Ucc1
        iz += S
        res[iz:iz+(I-1)*S] = (alphai*pii*Uni - mu*( Uni+ni*Unni ) 
            + xi*FN(K_,N)*pii*thetai - phi*theta1*Unni).flatten()
        iz += (I-1)*S
        res[iz:iz+S] = alpha1*pi1*Un1 + np.sum(rho*mu,0)*( Un1+n1*Unn1) + xi*FN(K_,N)*pi1*theta1\
         +np.sum(phi*thetai*rho,0)*Unn1
        iz += (I-1)*S
        
        #pdb.set_trace()
        res[iz:iz+S] = beta*ExiFK - xi-barlam+lambar
        return res
        
    def policy_residuals_end(self,K_,z):
        '''
        Compute the policy residuals 
        '''
        mu,rho = self.mu,self.rho
        I,S = len(alpha),len(P)
        #unpack policies and functions
        c1,ci,n1,ni,phi,xi,eta = getQuantities_end(z)
        #compute marginal utilities
        N = pi1*theta1*n1+np.sum(pii*thetai*ni,0)
        Uc1,Uci = Uc(c1,n1),Uc(ci,ni)
        Ucc1,Ucci = Ucc(c1,n1),Ucc(ci,ni)
        Un1,Uni = Un(c1,n1),Un(ci,ni)
        Unn1,Unni = Unn(c1,n1),Unn(ci,ni)
        #compute residuals
        res = np.zeros(len(z))
        res[:(I-1)*S] = (Un1*thetai*rho-Uni*theta1).flatten()
        iz = (I-1)*S
        res[iz:iz+S] = F(K_,N)-g-pi1*c1-np.sum(pii*ci,0)
        iz += S
        res[iz:iz+(I-1)*S] = (rho*Uc1 - Uci).flatten()
        iz += (I-1)*S
        res[iz:iz+(I-1)*S] = (alphai*pii*Uci - mu*(Uci+ci*Ucci) - pii*xi - eta*Ucci).flatten()
        iz += (I-1)*S
        res[iz:iz+S] = alpha1*pi1*Uc1 + np.sum(rho*mu,0)*( Uc1+c1*Ucc1 )-pi1*xi + np.sum(rho*eta,0)*Ucc1
        iz += S
        res[iz:iz+(I-1)*S] = (alphai*pii*Uni - mu*( Uni+ni*Unni ) 
            + xi*FN(K_,N)*pii*thetai - phi*theta1*Unni).flatten()
        iz += (I-1)*S
        res[iz:iz+S] = alpha1*pi1*Un1 + np.sum(rho*mu,0)*( Un1+n1*Unn1) + xi*FN(K_,N)*pi1*theta1\
         +np.sum(phi*thetai*rho,0)*Unn1
        return res
        
def solveProblem(mu,rho,Kgrid):
    '''
    Solves the problem for a given mu,rho using Kgrid
    '''
    print mu,rho
    Kbar = [Kgrid[0],Kgrid[-1]]
    I,S = len(alpha),len(P)
    interpolate1d = interpolator_factory(['spline'],[len(Kgrid)],[3])
    PRnew = iterate_on_policies(mu,rho,None,Kbar)
    PRs = np.vstack(map(PRnew,Kgrid))
    PRf = interpolate1d(Kgrid,PRs)
    diff = 1.
    while diff >1e-5:
         PRnew = iterate_on_policies(mu,rho,PRf,Kbar)
         PRs = np.vstack(map(lambda k:PRnew(np.array([k])),Kgrid))
         diff = np.amax(abs(PRs-PRf(Kgrid.reshape(-1,1)).T))
         PRf = interpolate1d(Kgrid,PRs)
         
    xf,xs = compute_x(mu,rho,Kgrid,PRf)
    DV = compute_DV(mu,rho,Kgrid,PRf)
    PRCM = []
    for s_ in range(S):
        PRCM.append([])
        for ik in range(len(Kgrid)):
            c1,ci,n1,ni,K,phi,xi,eta = getQuantities(PRs[ik])
            N = pi1*n1*theta1 + np.sum(pii*ni*thetai,0)
            Rk = FK(Kgrid[ik],N)
            PRCM[s_].append(np.hstack((
            c1,ci.flatten(),n1,ni.flatten(),Rk,xs[s_][ik],np.zeros(I-1),(rho*np.ones(S)).flatten(),
            K,(mu*np.ones(S)).flatten(),phi.flatten(),np.zeros(I-1),np.zeros(I-1),np.zeros(S),xi,eta.flatten(),
            DV[s_][ik]            
            )))
        PRCM[s_] = np.vstack(PRCM[s_])
    return PRCM
    
        
def compute_x(mu,rho,Kgrid,PRf):
    '''
    Computes x given the policy rules
    '''
    I,S = len(alpha),len(P)
    interpolate1d = interpolator_factory(['spline'],[len(Kgrid)],[3])
    c1f,cif,n1f,nif,Kf,phif,xif,etaf = getQuantities(PRf)
    xf = []
    for s in range(S):
        xf.append(interpolate1d(Kgrid,np.zeros((len(Kgrid),I-1))).reshape(((I-1),1)))
    xnew_old = [np.zeros((len(Kgrid),(I-1)))]*S
    diff = 1
    while diff > 1e-5:
        xnew = []
        for s_ in range(S):
            def xprime(k):
                xprime = np.zeros((I-1,S))
                for s in range(S):
                    Kprime = np.atleast_1d(Kf[s](k))
                    xprime[:,s] = xf[s](Kprime).flatten()
                return xprime
            xnew.append(np.zeros((len(Kgrid),I-1)))
            for ik in range(len(Kgrid)):
                k = np.array([Kgrid[ik]])
                c1,ci,n1,ni = c1f(k),cif(k),n1f(k),nif(k)
                xnew[-1][ik,:] = beta*(Uc(ci,ni)*ci+Un(ci,ni)*ni-rho*(Uc(c1,n1)*c1+Un(c1,n1)*n1)+xprime(k)).dot(P[s_,:])
        diff = 0.
        xf = []
        for s_ in range(S):
            diff = max(np.amax(np.abs(xnew[s_]-xnew_old[s_])),diff)
            xf.append(interpolate1d(Kgrid,xnew[s_]).reshape(((I-1),1)))
            xnew_old = xnew
    return xf,xnew
        
def compute_DV(mu,rho,Kgrid,PRf):
    '''
    Computes x given the policy rules
    '''
    I,S = len(alpha),len(P)
    interpolate1d = interpolator_factory(['spline'],[len(Kgrid)],[3])
    c1f,cif,n1f,nif,Kf,phif,xif,etaf = getQuantities(PRf)
    Vrhof = []
    
    for s in range(S):
        Vrhof.append(interpolate1d(Kgrid,np.zeros((len(Kgrid),I-1))).reshape(((I-1),1)))
    Vrho_old = [np.zeros((len(Kgrid),(I-1)))]*S
    diff = 1
    while diff > 1e-5:
        Vrho_new = []
        for s_ in range(S):
            def Vrho_prime(k):
                Vrho_prime = np.zeros((I-1,S))
                for s in range(S):
                    Kprime = np.atleast_1d(Kf[s](k))
                    Vrho_prime[:,s] = (Vrhof[s](Kprime)).flatten()
                return Vrho_prime
            Vrho_new.append(np.zeros((len(Kgrid),I-1)))
            for ik in range(len(Kgrid)):
                k = np.array([Kgrid[ik]])
                c1,n1,phi,eta = c1f(k),n1f(k),phif(k),etaf(k)
                Uc1,Un1 = Uc(c1,n1),Un(c1,n1)
                Vrho_new[-1][ik,:] = (phi*thetai*Un1 + eta*Uc1+beta*Vrho_prime(k)).dot(P[s_,:])
        diff = 0.
        Vrhof = []
        for s_ in range(S):
            diff = max(np.amax(np.abs(Vrho_new[s_]-Vrho_old[s_])),diff)
            Vrhof.append(interpolate1d(Kgrid,Vrho_new[s_]).reshape(((I-1),1)))
            Vrho_old = Vrho_new
            
    VK = []
    for s_ in range(S):
        VK.append(np.zeros((len(Kgrid),1)))
        for ik in range(len(Kgrid)):
            K_ = np.array([Kgrid[ik]])
            N = pi1*n1f(K_)*theta1+np.sum(pii*nif(K_)*thetai,0)
            VK[-1][ik] = P[s_,:].dot(FK(K_,N)*xif(K_))
    DV = []
    for s_ in range(S):
        DV.append(np.hstack((Vrho_new[s_],VK[s_])))
    return DV
    
        
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
    K = z[iz:iz+S]
    iz += S
    phi = z[iz:iz+(I-1)*S].reshape((-1,S))
    iz += (I-1)*S
    xi = z[iz:iz+S]
    iz += S
    eta = z[iz:iz+(I-1)*S].reshape((-1,S))
    return c1,ci,n1,ni,K,phi,xi,eta
    
def getQuantities_end(z):
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
    phi = z[iz:iz+(I-1)*S].reshape((-1,S))
    iz += (I-1)*S
    xi = z[iz:iz+S]
    iz += S
    eta = z[iz:iz+(I-1)*S].reshape((-1,S))
    return c1,ci,n1,ni,phi,xi,eta
    
def projectVariable(x,xbar):
    '''
    Projects a variable x onto the range [xbar[0],xbar[1]]
    '''
    fproj = np.vectorize(lambda x: max(min(x,xbar[1]),xbar[0]))
    return fproj(x)
    
    
def solveWerningProblem_parallel(Para,muGrid,rhoGrid,Kgrid,c):
    '''
    Solves the werning problem using IPython parallel
    '''
    #sets up some parrallel stuff
    v_lb = c.load_balanced_view()
    v_lb.block = True
    v = c[:]
    v.block = True
    #Setup
    calibrateFromParaStruct(Para)
    v.apply(calibrateFromParaStruct,Para)
    XCM = np.vstack(makeGrid_generic((muGrid,rhoGrid)))
    
    #solve the problem
    PRCMtemp = v_lb.map(lambda x: solveProblem(x[0][0],x[0][1],x[1]),zip(XCM,[Kgrid]*len(XCM)))
    PRs = []    
    S = len(P)
    for s_ in range(S):
        temp = []
        for ik in range(len(Kgrid)):
            for pr in PRCMtemp:
                temp.append(pr[s_][ik])
        PRs.append(np.vstack(temp))
    return PRs
        
    
    