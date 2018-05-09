# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:19:46 2018

@author: zl17868
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
import time
def cal_card(agents):
    sumcard = 0;  
    N =len(agents) 
    for i in range(N):
        sumcard = sumcard + len (agents[i])
    mean_card = sumcard/N
    return mean_card
def create_world(propsition_number):
    single_agent =set()# [set()]*an # use SET as for any agents
   # agents_temp = []
    agent_tuple = [] # belief as tuple(tuple is unchangeable)
    states = [0]*propsition_number #proposition as list 
    world = [] #A list of sets of tuple
    # initialise the agents
    agents_number = int(math.pow(2,propsition_number))
    for i in range(agents_number):
        basic = i+1
        for j in range (propsition_number):
            states[j] = basic%2
            basic = basic//2
        states_set = tuple(states)
        agent_tuple.append(states_set)
        single_agent = set (agent_tuple)
        #print (states_set)
        #print (single_agent)    
        agent_tuple.clear()
    #print (single_agent)
        world.append(single_agent)
#    world_set = world[0]
#    for k in range(agents_number):
#        world_set = world_set|world[k]

    return world
    

def random_initialise(agents_number, propsition_number,world):
    '''
    initialise the agents randomly (with a random number of beliefs)
    '''
    single_agent =set()# [set()]*an # use SET as for any agents
   # agents_temp = []
    agent_tuple = [] # belief as tuple(tuple is unchangeable)
    states = [0]*propsition_number #proposition as list 
    agents = [] #A list of sets of tuple
    indexlist = list(range(2**propsition_number))
#    print(indexlist)
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
    for num in range(len(agents)):
        num_blf = random.randint(1,int(math.pow(2,propsition_number)))
        agt = [agents[num]]
        random.shuffle(indexlist)
        k = 0
        while cal_card(agt) < num_blf :
#            print (agt)
#            print(cal_card(agt))
#            print (num_blf)
#            print(agents)
            agents[num] = agents[num]|world[indexlist[k]]
            k=k+1
            agt = [agents[num]]
   
            
    return agents

print(str(70))
start = time.clock()
agents = random_initialise(50, 5,create_world(5))
print(agents)
elapsed = (time.clock() - start)
print("Time used:",elapsed)