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

def initialise_agents(agents_number, propsition_number):
    '''
    initialise the agents randomly
    '''
    single_agent =set()# [set()]*an
   # agents_temp = []
    agent_tuple = []
    states = [0]*sn
    agents = []
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

def cal_card(agents):
    sumcard = 0;
    N =len(agents)
    for i in range(N):
        sumcard = sumcard + len (agents[i])
    mean_card = sumcard/N
    return mean_card

def cal_similarity(agents):
    #calculate similarity
    similarity = []
    simtotal = []
    an = len(agents)
    for i in range(an):
        for j in range (i+1,an):
            Num_inter =len(agents[i]&agents[j])
            Num_uni = len(agents[i]|agents[j])
            similarity.append((Num_inter/Num_uni))
        simtotal.append(similarity)

    return similarity

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
    return averagesim, cardinality

an = 100 #Number of agents
sn = 5  #Number of propsitions
N = 1000 # Times of iterations
T = 1
sim = []
card = []
AVEsim=[]
AVEcard = []
print(T)
#averagesim = [0]*T
#cardinality = [0]*T
agents = initialise_agents(an, sn);
for i in range (T):
    trans = copy.deepcopy(agents)
    #print (agents)
    (averagesim, cardinality) = iteration(trans,an,N)
    #print (agents)
    sim.append(averagesim)
    card.append(cardinality)
    #print (averagesim)
sumsim = [0]*len(averagesim)
sumcard = [0]*len(cardinality)
for i in range (T):

    sumsim = (np.sum([sumsim,sim[i]],axis = 0))
    sumcard =(np.sum([sumcard,card[i]],axis = 0))

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


print(an)
plt.figure(1)
plt.plot(AVEsim)
plt.show()
plt.figure(2)
plt.errorbar(index, AVEsim_f, yerr = stdsim_f, fmt ='-o',color = 'brown')
plt.show()
plt.figure(3)
plt.plot(AVEcard)
plt.show()
plt.figure(4)
plt.errorbar(index, AVEcard_f, yerr = stdcard_f, fmt ='-o',color = 'brown')
plt.show()
