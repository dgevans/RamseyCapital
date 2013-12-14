# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:35:31 2013

@author: dgevans
"""
from ctypes import c_double
import ctypes
from nag4py.util import NagError, Nag_Comm, Nag_NoPrint, Nag_FALSE, SET_FAIL
from nag4py.c05 import c05qbc, NAG_C05QBC_FUN
import numpy as np

class result(object):
    def __init__(self,success,x):
        self.success = success
        self.x = x

def root(f,x0):
    '''
    Finds the root 
    '''
    def py_fun(n, xp, fvecp, comm, iflag):
        ArrayType = c_double*n
        addr = ctypes.addressof(fvecp.contents)
        fvec = np.frombuffer(ArrayType.from_address(addr))
        x = np.fromiter(xp,np.float,count = n)
        fvec[:] = f(x)[:]
        
        
    comm = Nag_Comm()
    c_fun = NAG_C05QBC_FUN(py_fun)
    n = len(x0)
    fvec = f(x0)
    
    fail = NagError()
    SET_FAIL(fail)
    
    c05qbc(c_fun,n,x0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
           fvec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),1e-10,comm,fail)
    if fail.code > 0:
        return result(False,x0)
    return result(True,x0)
    