# -*- coding: utf-8 -*-
"""
Created on Fri Jun 1 00:13:19 2018

@author: liuzx
"""
import sys
import math
import numpy as np
def Hamming(x,y):#x,y are lists
    hamming = []
    for a in x:
        for b in y:
            a = np.array(a)
            b= np.array(b)
            #print(a)
            #print(b)
            vec = (a ^ b)
            #print (vec)
    hamming = sum (vec)
    return hamming#, hamming

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

def find_neighbours(agent,radius,village):
	#agent is who looks for its neighbours
    #radius is the range in which other beliefs are defined as neighbours
    #village is the 'world' where we look for the neoghbours
    neighbours = agent #[]
    for i in range(len(village)):
    	if Hamming(agent,village[i]) <= radius :
    		#neighbours.append(village[i])#got several sets
    		neighbours=neighbours|village[i]

    return neighbours


#print('Please input your belief')
belief = {(1,0,0,1)}#input()
village= create_world(4)
print(find_neighbours(belief, 2, village))

