# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:32:00 2013

@author: dgevans
"""
from numpy import *
import cPickle
from IPython.parallel import Client
c = Client()
v = c[:]
with v.sync_imports():
    import WerningProblem as WP
    import PlannersProblem as PP
    import utilities
    import calibration

#PRs,Para = cPickle.load(file('baseline2.dat','r')) 
Para = calibration.baseline2
S = len(Para.P)   


WP.calibrateFromParaStruct(Para)
PP.calibrateFromParaStruct(Para)


muGrid = linspace(0.55,0.75,10)
rhoGrid = linspace(1.5,1.7,10)
Kgrid = linspace(1.,2.5,10)


XCM = vstack(utilities.makeGrid_generic((muGrid,rhoGrid)))
X = vstack(utilities.makeGrid_generic((muGrid,rhoGrid,Kgrid)))
Para.X = X

PRs = WP.solveWerningProblem_parallel(Para,muGrid,rhoGrid,Kgrid,c)

mubar = [muGrid[0],muGrid[-1]]
rhobar = [rhoGrid[0],rhoGrid[-1]]
#PF = PP.solvePlannersProblemIID_parallel(PF,Para,X,mubar,rhobar,c)
interpolate3d_lin = utilities.interpolator_factory(['spline']*3,[10,10,10],[1]*3)
PRs_old = []
PF = []
for s_ in range(S):
    PRs_old.append(PRs[s_])
    PF.append(interpolate3d_lin(X,PRs[s_]))
interpolate3d = utilities.interpolator_factory(['spline']*3,[10,10,10],[3]*3)
diff = 1.
while diff > 1e-5:
    PRs = PP.iteratePlannersProblemIID_parallel(PF,Para,X,mubar,rhobar,c) 
    diff = amax(abs((PRs[0]-PRs_old[0])))
    rel_diff = amax(abs((PRs[0]-PRs_old[0])/PRs[0]))
    print "diff: ",diff,rel_diff
    PF = []
    for s_ in range(len(Para.P)):
        if diff<1:
            PF.append(interpolate3d(X,PRs[s_]))
        else:
            PF.append(interpolate3d_lin(X,PRs[s_]))
        PRs_old[s_] = PRs[s_]