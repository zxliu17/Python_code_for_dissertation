# -*- coding: utf-8 -*-
"""
Created on Wed May  9 00:13:19 2018

@author: liuzx
"""

import math
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
print(create_world(4))
