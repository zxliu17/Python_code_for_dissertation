# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:13:27 2018

@author: liuzx
"""

import numpy as np

def hammingdis(x, y):  
    vec = (x ^ y)
    return sum (vec)  
    
an = 50 #Number of agents
sn = 5  #Number of states
N = 5000 # Times of iterations 
agents = np.random.randint(0,2,(an,sn)) #Set every agents to some state randomly

for i in range(an):
    if(sum(agents[i])== 0):
         print(sum(agents[i]))
         #print("pre")
         #print(agents[i])
         agents[i]= np.random.randint(0,2,(sn))
         #print("after")
         #print(agents[i])
         #print(i)
         i=i-1
         #print(i)
      
print(agents)

iteration=0 # iteration time count

while iteration < N:
    iteration =iteration +1 
   # print (iteration)
    index1 = np.random.randint(0,an)
    index2 = np.random.randint(0,an)
    t = agents [index1]
    s = agents [index2]
    distance = hammingdis(s,t) # check if overlap exists
    if (distance == sn) : 
           agents [index1] =s |t #Union if Hamming distance equals to sn 
    #    print (agents [index1])
    else:
           agents [index2]= s & t #intersect if not
     #   print (agents [index2])
    
    variance = np.var(agents, axis = 0)
    #print (variance)
print(agents)

