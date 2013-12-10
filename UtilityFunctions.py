# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:47:58 2013

@author: dgevans
"""
import numpy as np

class CRRA(object):
    '''
    Holds the CRRA utility funciton
    '''
    def __init__(self,sigma,gamma):
        '''
        Initializes storing sigma and gamma
        '''
        self.sigma = sigma
        self.gamma = gamma
    def U(self,c,n):
        '''
        Utility
        '''
        U = 0
        if self.sigma == 0:
            U = np.log(c)
        else:
            U = (c**(1-self.sigma)-1)/(1-self.sigma)
        return U - n**(1+self.gamma)/(1+self.gamma)
        
    def Uc(self,c,n):
        '''
        Uc
        '''
        return c**(-self.sigma)
    def Ucc(self,c,n):
        '''
        Ucc
        '''
        return -self.sigma*c**(-self.sigma-1)
    def Un(self,c,n):
        '''
        Ul
        '''
        return -n**(self.gamma)
    def Unn(self,c,n):
        '''
        Ull
        '''
        return -self.gamma*n**(self.gamma-1.)
