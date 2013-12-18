# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:05:18 2013

@author: dgevans
"""
import numpy as np
from cpp_interpolator import interpolate
from cpp_interpolator import interpolate_INFO
from Spline import Spline

class interpolate_wrapper(object):
    '''
    Wrapper to interpolate vector function
    '''
    def __init__(self,F):
        '''
        Inits with array of interpolated functions
        '''
        self.F = F
    def __getitem__(self,index):
        '''
        Uses square brakets operator
        '''
        return interpolate_wrapper(self.F[index])
    def reshape(self,*args):
        '''
        Reshapes F
        '''
        self.F = self.F.reshape(*args)
        return self
    def __len__(self):
        '''
        return length
        '''
        return len(self.F)
    def __call__(self,X):
        '''
        Evaluates F at X for each element of F, keeping track of the shape of F
        '''
        if X.ndim == 1 or len(X) == 1:
            fhat = np.vectorize(lambda f: float(f(X)),otypes=[np.ndarray])
        else:
            fhat = np.vectorize(lambda f: f(X).flatten(),otypes=[np.ndarray])
        return np.array(fhat(self.F).tolist())

class interpolator_factory(object):
    '''
    Generates an interpolator factory which will interpolate vector functions
    '''
    def __init__(self,types,orders,k):
        '''
        Inits with types, orders and k
        '''
        self.INFO = interpolate_INFO(types,orders,k)
        self.k = k
        
    def __call__(self,X,Fs):
        '''
        Interpolates function given function values Fs at domain X
        '''
        n,m = Fs.shape
        F = []
        for i in range(m):
            #F.append(interpolate(X,Fs[:,i],self.INFO))
            F.append(Spline(X,Fs[:,i],self.k))
        return interpolate_wrapper(np.array(F))
        
class splineExtender(object):
    '''
    Extends spline linearly outside boundry
    '''
    def __init__(self,X,y,k):
        '''
        
        '''
        self.F = Spline(X,y,k)
        self.Xbar = np.vstack((np.amin(X,0),np.amax(X,0)))
        if X.ndim == 1:
            self.N = 1
        else:
            self.N = X.shape[1]
        
    def __call__(self,X):
        '''
        Evaluate spline
        '''
        N = self.N
        X = np.atleast_2d(X)
        if X.shape[0] == self.N:
            X = X.T
        Xtilde = np.zeros(X.shape)
        for j in range(N):
            Xtilde[:,j] = projectVariable(X[:,j],self.Xbar[:,j])
        f = self.F(Xtilde)
        Xdiff = X - Xtilde
        D = np.eye(N,dtype=int)
        for i in range(len(Xdiff)):
            for j in range(N):
                if Xdiff[i,j] >0:
                    f[i] += self.F(Xtilde[i,:],D[j,:])*Xdiff[i,j]
        return f     
            
        
def makeGrid_generic(x):
    '''
    Makes a grid to interpolate on from list of x's.
    '''
    N = 1
    n = []
    n.append(N)
    for i in range(0,len(x)):
        N *= len(x[i])
        n.append(N)
    X =[]
    for i in range(0,N):
        temp = i
        temp_X = []
        for j in range(len(x)-1,-1,-1):
            temp_X.append(x[j][temp/n[j]])
            temp %= n[j]
        temp_X.reverse()
        X.append(temp_X)
    return X
    
def projectVariable(x,xbar):
    '''
    Projects a variable x onto the range [xbar[0],xbar[1]]
    '''
    fproj = np.vectorize(lambda x: max(min(x,xbar[1]),xbar[0]))
    return fproj(x)
    
    