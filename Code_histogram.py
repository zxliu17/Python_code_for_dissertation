# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:13:27 2018

@author: liuzx
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
'''Function for hamming distance'''
def hammingdis(x, y):  
    vec = (x ^ y)
    return sum (vec)  #return value is a scalar

def initialise_agents(agents_number, propsition_number):
    '''
    initialise the agents randomly (with a single belief)
    '''
    single_agent =set()# [set()]*an # use SET as for any agents
   # agents_temp = []
    agent_tuple = [] # belief as tuple(tuple is unchangeable)
    states = [0]*propsition_number #proposition as list 
    agents = [] #A list of sets of tuple
    # initialise the agents
    for i in range(agents_number):
        for j in range (propsition_number):
            states[j] = (random.randint(0,1))
        states_set = tuple(states)
        agent_tuple.append(states_set)
        single_agent = set (agent_tuple)
        #print (states_set)
        #print (single_agent)    
        agent_tuple.clear()
    #print (single_agent)
        agents.append(single_agent)
    return agents
'''compute the average cardinality of all agents'''
def cal_card(agents):
    sumcard = 0;  
    N =len(agents) 
    for i in range(N):
        sumcard = sumcard + len (agents[i])
    mean_card = sumcard/N
    return mean_card
''' transfer binary string to DEC number'''
def trans2dec(list_set_of_tuple):
    dec= []
    for index in range(1):# range(len(list_set_of_tuple)):
#    print(set_of_tuple)
        for x in list_set_of_tuple[index]:
            a=0
            for i in range(len(x)):
                a = a+x[i]*math.pow(2,len(x)-i-1)
#            print(a)
            dec.append(a)
    
    return dec

'''compute the average simlarity of all agents'''
def cal_similarity(agents):
    #calculate similarity
    similarity = []
    simtotal = []
    an = len(agents)
    for i in range(an):
        for j in range (i,an):
            Num_inter =len(agents[i]&agents[j])
            Num_uni = len(agents[i]|agents[j])
            similarity.append((Num_inter/Num_uni))
        simtotal.append(similarity)
    
    return similarity 
''' one time for combine beliefs'''
def iteration(agents,agent_number, iteration_times):
    
    an = agent_number
    #sn = proposition_number
    N = iteration_times
    averagesim = [];
    iteration=0 # iteration time count
    cardinality  = [];
    while iteration < N:
        iteration =iteration +1 
        # print (iteration)
        index1 = random.randint(0,an-1)
        index2 = random.randint(0,an-1)
        #t = agents [index1]
        #s = agents [index2]
        Intersection = agents[index1]&agents[index2]
        Union = agents[index1]|agents[index2]
    #distance = hammingdis(s,t) # check if overlap exists
        if (Intersection == set()) : 
               agents [index1] =Union 
               agents [index2] =Union

        else:
               agents [index1]=Intersection#intersect if not
               agents [index2]=Intersection
   
        mean_card  = cal_card(agents)
        cardinality.append(mean_card)
        similarity = cal_similarity(agents)
        averagesim.append(sum(similarity)/len(similarity))
    '''
    print (averagesim)  
    plt.figure(1)
    plt.plot(averagesim)#, sta)
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Stimulus')
    #plt.title('Spike-Triggered Average')
    #plt.savefig('C:\\Users\\liuzx\\Spyderpro\\CW2\\Computational-Neuroscience-coursework2\\Spike_Triggered_Average(Q3)')
    plt.show()
    plt.figure(2)
    plt.plot(cardinality)
    plt.show()
    #print (similarity)
    #print (simtotal)
    '''
    return averagesim, cardinality , agents
    
an = 50 #Number of agents
sn = 3  #Number of propsitions
N = 500 # Times of iterations 
T = 300
sim = []
card = []
AVEsim=[]
AVEcard = []
convergePos = []
#averagesim = [0]*T
#cardinality = [0]*T
#agents = initialise_agents(an, sn);

for i in range (T):
    agents = initialise_agents(an, sn);
    trans = copy.deepcopy(agents)
    #print (agents)
    (averagesim, cardinality, store) = iteration(trans,an,N)
    dec = trans2dec(store)
    pos = copy.deepcopy(dec[0])
    #print (agents)
    convergePos.append(pos)
    sim.append(averagesim)
    card.append(cardinality)
    #print (averagesim)
sumsim = [0]*len(averagesim)
sumcard = [0]*len(cardinality)
countagt = [0]*int(math.pow(2,sn))
for i in range (T):
    
    sumsim = (np.sum([sumsim,sim[i]],axis = 0))
    sumcard =(np.sum([sumcard,card[i]],axis = 0))
    countagt[int(convergePos[i])]=countagt[int(convergePos[i])]+1

xaxis = np.arange(1, len(countagt)+1)    
AVEsim=sumsim/T
AVEcard=sumcard/T
stdsim =  np.std(sim,axis=0)
stdcard =  np.std(card,axis=0)
stdsim_f = []
stdcard_f= []
AVEcard_f = []
AVEsim_f = []
index = []
j=0
while j < len(stdsim):

    index.append(j)
    stdsim_f.append(stdsim[j])
    stdcard_f.append(stdcard[j])
    AVEcard_f.append(AVEcard[j])
    AVEsim_f.append(AVEsim[j])
    j = j+50
    


plt.figure(1)
plt.plot(AVEsim)
plt.show()
plt.figure(2)
plt.plot(AVEsim,color = 'brown')
plt.errorbar(index, AVEsim_f, yerr = stdsim_f, fmt ='o',color = 'brown')
plt.show()
plt.figure(3)
plt.plot(AVEcard)
plt.show()
plt.figure(4)
plt.plot(AVEcard,color = 'brown')
plt.errorbar(index, AVEcard_f, yerr = stdcard_f, fmt ='o',color = 'brown')
plt.show()
plt.figure(5)
plt.bar(xaxis,countagt,color = 'black',width = 0.4) 
plt.title("Times of covergence") 
plt.show()


