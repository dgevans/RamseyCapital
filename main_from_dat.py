# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:13:46 2014

@author: dgevans
"""

from numpy import *
import cPickle
import itertools
import WerningProblem as WP
import PlannersProblem as PP
import utilities
import calibration

PRs,Para = cPickle.load(file('CMFlipped.dat','r'))  

WP.calibrateFromParaStruct(Para)
PP.calibrateFromParaStruct(Para)

X = Para.X
muGrid = unique(X[:,0])
rhoGrid = unique(X[:,1])
Kgrid = unique(X[:,2])
S = len(Para.P)

interpolate3d = utilities.interpolator_factory(['spline']*3,[10,10,10],[3]*3)
PF = [interpolate3d(X,PRs[0])]*S