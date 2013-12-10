# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:32:00 2013

@author: dgevans
"""
from numpy import *
from IPython.parallel import Client
c = Client()
v = c[:]
with v.sync_imports():
    import WerningProblem as WP
    import PlannersProblem as PP
    import utilities
    import calibration
    
Para = calibration.baseline

WP.calibrateFromParaStruct(Para)
PP.calibrateFromParaStruct(Para)


muGrid = linspace(0.5,0.8,10)
rhoGrid = linspace(1.5,1.9,10)
Kgrid = linspace(1.,2.5,10)

PF = WP.solveWerningProblem_parallel(Para,muGrid,rhoGrid,Kgrid)

mubar = [muGrid[0],muGrid[-1]]
rhobar = [rhoGrid[0],rhoGrid[-1]]
PF = PP.solvePlannersProblemIID_parallel(PF,Para,X,mubar,rhobar)