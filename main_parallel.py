# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:32:00 2013

@author: dgevans
"""
from numpy import *
import cPickle
import itertools
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
theta1 = Para.theta
theta0 = Para.theta.dot(Para.P[0,:]).reshape(-1,1)

alpha = linspace(0,1,10)
a = alpha[1]
Para.theta = a*theta1+(1-a)*theta0

S = len(Para.P)   


WP.calibrateFromParaStruct(Para)
PP.calibrateFromParaStruct(Para)


muGrid = linspace(0.5,0.8,10)
rhoGrid = linspace(1.6,2.,10)
Kgrid = linspace(1.,2.5,10)


XCM = vstack(utilities.makeGrid_generic((muGrid,rhoGrid)))
X = vstack(utilities.makeGrid_generic((muGrid,rhoGrid,Kgrid)))
Para.X = X

PRs = WP.solveWerningProblem_parallel(Para,muGrid,rhoGrid,Kgrid,c)

mubar = [muGrid[0],muGrid[-1]]
rhobar = [rhoGrid[0],rhoGrid[-1]]
#PF = PP.solvePlannersProblemIID_parallel(PF,Para,X,mubar,rhobar,c)
interpolate3d_lin = utilities.interpolator_factory(['spline']*3,[10,10,10],[3]*3)
PRs_old = []
PF = []
for s_ in range(S):
    PRs_old.append(PRs[s_])
    PF.append(interpolate3d_lin(X,PRs[s_]))
interpolate3d = utilities.interpolator_factory(['spline']*3,[10,10,10],[3]*3)
for a in alpha[1:]:
    Para.theta = a*theta1+(1-a)*theta0
    print "new theta:",Para.theta
    WP.calibrateFromParaStruct(Para)
    PP.calibrateFromParaStruct(Para)
    diff = 1.
    while diff > 1e-5:
        policies = PP.iteratePlannersProblemIID_parallel(PF,Para,X,mubar,rhobar,c)
        try:
            PRs = [np.vstack(policies)]*S
        except:
            PRnew = PP.iterate_on_policies(PF,mubar,rhobar)
            PP.findFailedRoots(PRnew,policies,X,0)
            solved = filter(lambda x: x[1][0] != None,enumerate(itertools.izip(policies,X)))
            iSolved,PRXsolved = zip(*solved)
            PRsolved,Xsolved = map(np.vstack,zip(*PRXsolved))
            F = interpolate3d(Xsolved,PRsolved)
            PRs = [F(X).T]*S
        diff = amax(abs((PRs[0]-PRs_old[0])))
        rel_diff = amax(abs((PRs[0]-PRs_old[0])/PRs[0]))
        print "diff: ",diff,rel_diff
        PF = []
        for s_ in range(len(Para.P)):
            if diff<1.:
                PF.append(interpolate3d(X,PRs[s_]))
            else:
                PF.append(interpolate3d(X,PRs[s_]))
            PRs_old[s_] = PRs[s_]