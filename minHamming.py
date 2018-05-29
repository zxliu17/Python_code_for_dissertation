# -*- coding: utf-8 -*-
"""
Created on Tue May 15 21:20:35 2018

@author: liuzx
"""
import numpy as np
def minHamming(x,y):#x,y are lists
    hamming = []
    for a in x:
        for b in y:
            a = np.array(a)
            b= np.array(b)
            #print(a)
            #print(b)
            vec = (a ^ b)
            print (vec)
            hamming.append(sum (vec))  
    return min(hamming)/len(a)#, hamming


x = {(1,0,1,0),(1,0,0,0)}
y = {(1,1,1,0),(1,1,1,1)}


print (minHamming(x,y))