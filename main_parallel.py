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
Para = calibration.baseline3
theta1 = Para.theta
theta0 = Para.theta.dot(Para.P[0,:]).reshape(-1,1)

alpha = linspace(0,1,6)
a = alpha[1]
Para.theta = a*theta1+(1-a)*theta0

S = len(Para.P)   


WP.calibrateFromParaStruct(Para)
PP.calibrateFromParaStruct(Para)


muGrid = linspace(0.4,0.75,10)
rhoGrid = linspace(1.65,2.,10)
Kgrid = linspace(3.,5.,10)


XCM = vstack(utilities.makeGrid_generic((muGrid,rhoGrid)))
X = vstack(utilities.makeGrid_generic((muGrid,rhoGrid,Kgrid)))
Para.X = X

PRs = WP.solveWerningProblem_parallel(Para,muGrid,rhoGrid,Kgrid,c)
fout = file('progressCM.dat','w')
cPickle.dump((PRs,Para),fout)
fout.close()

mubar = [muGrid[0],muGrid[-1]]
rhobar = [rhoGrid[0],rhoGrid[-1]]
#PRs,Para = cPickle.load(file('progress.dat','r'))  
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
    diffbar = 1e-4
    if a == alpha[-1]:
        diffbar = 1e-5
    while diff > diffbar:
        policies = PP.iteratePlannersProblemIID_parallel(PF,Para,X,mubar,rhobar,c)
        try:
            PRs = [vstack(policies)]*S
        except:
            PRnew = PP.iterate_on_policies(PF,mubar,rhobar)
            PP.findFailedRoots(PRnew,policies,X,0)
            solved = filter(lambda x: x[1][0] != None,enumerate(itertools.izip(policies,X)))
            iSolved,PRXsolved = zip(*solved)
            PRsolved,Xsolved = map(vstack,zip(*PRXsolved))
            F = interpolate3d(Xsolved,PRsolved)
            PRs = [F(X).T]*S
        diff = amax(abs((PP.getXDV(PRs[0])-PP.getXDV(PRs_old[0]))))
        rel_diff = amax(abs((PP.getXDV(PRs[0])-PP.getXDV(PRs_old[0]))/PP.getXDV(PRs[0])))
        print "diff: ",diff,rel_diff
        PF = []
        for s_ in range(len(Para.P)):
            if diff<1.:
                PF.append(interpolate3d(X,PRs[s_]))
            else:
                PF.append(interpolate3d(X,PRs[s_]))
            PRs_old[s_] = PRs[s_]
    fout = file('progress.dat','w')
    cPickle.dump((PRs,Para),fout)
    fout.close()