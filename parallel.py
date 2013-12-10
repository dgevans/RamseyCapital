# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:51:13 2013

@author: dgevans
"""
import utilities

class testClass(object):
    def __init__(self,b):
        self.b = b
    def __call__(self):
        return self.b
a = 3.

def f():
    return a
    
def set_a(a_):
    global a
    a = a_
    
def test():
    from IPython.parallel import Client
    from IPython.parallel import Reference
    c = Client()
    v = c[:]
    v.block = True   
    v['F'] = testClass(4.)    
    rF = Reference('F')
    
    return v.apply(rF)