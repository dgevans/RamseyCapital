# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:01:26 2013

@author: dgevans
"""
import UtilityFunctions
import ProductionFunctions
import numpy as np
class parameters(object):
    pass

#baseline test
thetabar =np.array([0.75,0.25]).reshape((-1,1))
tfpshock = np.array([1.05,0.95])
ineq_shock = np.array([[1.1,1.],[0.9,1.]])
Ptfp = np.ones((2,2))/2.
Pineq = [np.array([[0.7,0.3],[0.7,0.3]]),np.array([[0.3,0.7],[0.3,0.7]])]
baseline = parameters()

baseline.Ufunc = UtilityFunctions.CRRA(2.,2.)
baseline.Ffunc = ProductionFunctions.CobbDouglass(0.3,0.05)
baseline.theta = thetabar*np.kron(tfpshock,ineq_shock)
baseline.P = np.vstack((np.kron(Ptfp[0,:],Pineq[0]),np.kron(Ptfp[1,:],Pineq[1])))
baseline.beta = 0.95
baseline.g = 0.17*np.ones(len(baseline.P))
baseline.alpha = np.array([0.5,0.5])
baseline.pi = np.array([1.,1.])
