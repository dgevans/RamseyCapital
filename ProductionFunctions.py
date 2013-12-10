# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:54:27 2013

@author: dgevans
"""

class CobbDouglass(object):
    '''
    Holds the revelent derivatives of the Cobb-Douglass production function
    '''
    def __init__(self,a,delta):
        '''
        Initializes with weight on capital
        '''
        self.a = a
        self.delta = delta
    def F(self,K,N):
        '''
        F
        '''
        return N**(1-self.a)*K**(self.a) + (1.-self.delta)*K
    def FK(self,K,N):
        '''
        F_K
        '''
        return self.a *(N/K)**(1-self.a) + 1. - self.delta
    def FN(self,K,N):
        '''
        F_N
        '''
        return (1.-self.a) * (K/N)**(self.a)
    def FKK(self,K,N):
        '''
        F_KK
        '''
        return (self.a-1)*self.FK(K,N)/K
    def FKN(self,K,N):
        '''
        F_{KN}
        '''
        return (1-self.a)*self.FK(K,N)/N