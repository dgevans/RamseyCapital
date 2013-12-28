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

#baseline2 test
thetabar =np.array([0.75,0.25]).reshape((-1,1))
tfpshock = np.array([1.02,0.98])
ineq_shock = np.array([[1.05,1.],[0.95,1.]])
Ptfp = np.ones((2,2))/2.
Pineq = [np.array([[0.3,0.7],[0.3,0.7]]),np.array([[0.7,0.3],[0.7,0.3]])]
baseline2 = parameters()

baseline2.Ufunc = UtilityFunctions.CRRA(2.,2.)
baseline2.Ffunc = ProductionFunctions.CobbDouglass(0.3,0.05)
baseline2.theta = thetabar*np.kron(tfpshock,ineq_shock)
baseline2.P = np.vstack((np.kron(Ptfp[0,:],Pineq[0]),np.kron(Ptfp[1,:],Pineq[1])))
baseline2.beta = 0.95
baseline2.g = 0.17*np.ones(len(baseline.P))
baseline2.alpha = np.array([0.5,0.5])
baseline2.pi = np.array([1.,1.])

#baseline2 test
thetabar =np.array([0.75,0.25]).reshape((-1,1))
tfpshock = np.array([1.02,0.98])
ineq_shock = np.array([[1.05,1.],[0.95,1.]])
Ptfp = np.ones((2,2))/2.
Pineq = [np.array([[0.7,0.3],[0.7,0.3]]),np.array([[0.3,0.7],[0.3,0.7]])]
baseline3 = parameters()

baseline3.Ufunc = UtilityFunctions.CRRA(2.,2.)
baseline3.Ffunc = ProductionFunctions.CobbDouglass(0.3,0.05)
baseline3.theta = thetabar*np.kron(tfpshock,ineq_shock)
baseline3.P = np.vstack((np.kron(Ptfp[0,:],Pineq[0]),np.kron(Ptfp[1,:],Pineq[1])))
baseline3.beta = 0.95
baseline3.g = 0.17*np.ones(len(baseline.P))
baseline3.alpha = np.array([0.5,0.5])
baseline3.pi = np.array([1.,1.])
