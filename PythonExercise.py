# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:13:27 2018

@author: liuzx
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import copy


def hammingdis(x, y):  
    vec = (x ^ y)
    return sum (vec)  
    
an = 100 #Number of agents
sn = 5  #Number of states
N = 1000 # Times of iterations 

'''
initialise the agents randomly
'''
single_agent =set()# [set()]*an
agents_temp = []
agent_tuple = []
states = [0]*sn
agents = []
# initialise the agents
for i in range(an):
    for j in range (sn):
        states[j] = (random.randint(0,1))
    states_set = tuple(states)
    agent_tuple.append(states_set)
    single_agent = set (agent_tuple)
    #print (states_set)
    #print (single_agent)    
    agent_tuple.clear()
    #print (single_agent)
    agents.append(single_agent)

#print (agents)
'''print (len(agents))
print (agents[0]&agents[1])

print (agents[0]|agents[1])'''

averagesim = [];
iteration=0 # iteration time count
length = [];
while iteration < N:
    iteration =iteration +1 
   # print (iteration)
    index1 = random.randint(0,an-1)
    index2 = random.randint(0,an-1)
    t = agents [index1]
    s = agents [index2]
    Intersection = agents[index1]&agents[index2]
    Union = agents[index1]|agents[index2]
    #distance = hammingdis(s,t) # check if overlap exists
    if (Intersection == set()) : 
           agents [index1] =Union #Union if Hamming distance equals to sn 
           agents [index2] =Union
           #print (agents [index1])
           #print (1)
    else:
           agents [index1]=Intersection#intersect if not
           agents [index2]=Intersection
           #print (agents [index2])
    #print (variance)
#print(agents)
    sumlength = 0;  
    for i in range(an):
        sumlength = sumlength + len (agents[i])
        
    length.append(sumlength/an);
        
    similarity=[]
    simtotal = []
#calculate similarity
    Num = int(an*(an-1)/2)-1
    for i in range(an):
        for j in range (i,an):
            Num_inter =len(agents[i]&agents[j])
            Num_uni = len(agents[i]|agents[j])
            similarity.append((Num_inter/Num_uni))
        simtotal.append(similarity)
    averagesim.append(sum(similarity)/len(similarity))
    #print (Num_inter)
    #print (Num_uni)
print (averagesim)  
plt.plot(averagesim)#, sta)
#plt.xlabel('Time (ms)')
#plt.ylabel('Stimulus')
#plt.title('Spike-Triggered Average')
#plt.savefig('C:\\Users\\liuzx\\Spyderpro\\CW2\\Computational-Neuroscience-coursework2\\Spike_Triggered_Average(Q3)')
plt.show()
plt.plot(length)
plt.show()
#print (similarity)
#print (simtotal)




