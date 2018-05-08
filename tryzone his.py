# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 01:57:25 2018

@author: liuzx
"""
import numpy as np
import random
import math
#a = [(1,2,3,4),(1,3,3,3)]
#b = [(2,3,4,6),(1,2,3,4)]
'''
a=[]
b=[]
agents = []
for i in range(an):
    agents.append(np.random.randint(0,2,(1,4)))
    a[i] = agents
print(a)
print(b)
print (set(a).intersection(set(b)))
print (set(a).union(set(b)))
c = np.intersect1d(a, b)
d = np.union1d(a ,b)
print(c)
print(d)
'''
'''
s = set([(1,2,3),(99,98,87)])
for item in s:
        print(item)
        
for i in enumerate(s):
        print (i)
'''
'''
a = tuple([1,2,4,3])
c= tuple([2,3,4,5])
print(a)
b = set()
b.add(a)
b.add(c)
print (b)
'''
'''

single_agent.add(states_set) #s.add(states_set) #Set every agents to some state randomly
    agents_temp.append(single_agent)
    agents = copy.deepcopy(agents_temp)
    single_agent.remove(states_set)
    
print(single_agent)
print(agents)
    #t1 = set(frozenset(i) for i in t)

iteration=0 # iteration time count

while iteration < N:
    iteration =iteration +1 
   # print (iteration)
    index1 = random.randint(0,an-1)
    index2 = random.randint(0,an-1)
    t = agents [index1]
    s = agents [index2]
    #distance = hammingdis(s,t) # check if overlap exists
    if (a==0) : 
           agents [index1] =np.union1d(s ,t) #Union if Hamming distance equals to sn 
           agents [index1] =np.union1d(s ,t)
    #    print (agents [index1])
    else:
           agents [index2]= np.intersect1d(s, t, assume_unique=True) #intersect if not
           agents [index2]= np.intersect1d(s, t, assume_unique=True)
     #   print (agents [index2])
    
    variance = np.var(agents, axis = 0)
    #print (variance)
print(agents)
    
an = 3 #Number of agents
sn = 4  #Number of states
N = 5000 # Times of iterations 
single_agent =set()# [set()]*an
agents_temp = []
agents = []
states = set()
# initialise the agents
for i in range(an):
    for j in range (sn):
        states.add(random.randint(0,1))
    agents_temp.append(states)
    #agent = frozenset(states_set) for k in sn
    print(i)
    print(states)
    print(agents_temp)
    #states=[]
print(agents_temp)
'''

#a = [[1,23,4],[1,4,3]]
#b = [2,2,4]
#c = [0]*3
#
#print(sum(a[]))
import numpy as np
import matplotlib.pyplot as plt

def trans2dec(set_of_tuple):
    dec= []
    b=0
#    print(set_of_tuple)
    for x in set_of_tuple:
        b=b+1
        a=0
        print(b)
        for i in range(len(x)):
            a = a+x[i]*math.pow(2,len(x)-i-1)
#            print(a)
        dec.append(a)
    
    return dec
def cal_mode(a):
    counts = np.bincount(a)
    return  np.argmax(counts)           
    
    
hui = {(1,0,0,1),(0,1,0,0),(0,1,0,1),(0,1,1,1)}
#result = trans2dec(hui)
#print (result[1])
#
#print (cal_mode(result))
result = [1,2,1,4,5,5,5]
xstcick = np.arange(1, len(result)+1)
plt.bar(result,xstcick,color = 'black',width = 0.4) 
plt.title("histogram") 
plt.show()

