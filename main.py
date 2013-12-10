# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:38:36 2013

@author: dgevans
"""

import WerningProblem as WP
import PlannersProblem as PP
import utilities
from numpy import *
import calibration

Para = calibration.baseline

WP.calibrateFromParaStruct(Para)
PP.calibrateFromParaStruct(Para)


muGrid = linspace(0.5,0.8,10)
rhoGrid = linspace(1.5,1.9,10)
Kgrid = linspace(1.,2.5,10)

XCM = vstack(utilities.makeGrid_generic((muGrid,rhoGrid)))
X = vstack(utilities.makeGrid_generic((muGrid,rhoGrid,Kgrid)))

Para.X = X

PRCMtemp = []
for mu,rho in XCM:
    PRCMtemp.append(WP.solveProblem(mu,rho,Kgrid))
    print mu,rho

PRCM = []
S = len(WP.P)
for s_ in range(S):
    temp = []
    for ik in range(len(Kgrid)):
        for pr in PRCMtemp:
            temp.append(pr[s_][ik])
    PRCM.append(vstack(temp))
    