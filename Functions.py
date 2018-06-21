'''
Functions needed in the project
'''
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
#import time
#import pickle
'''Function for max and min hamming distance'''
def maxHamming(x,y):#x,y are lists
    hamming = []
    for a in x:
        for b in y:
            a = np.array(a)
            b= np.array(b)
            #print(a)
            #print(b)
            vec = (a ^ b)
            #print (vec)
            hamming.append(sum (vec))
    return max(hamming)/len(a)#, hamming

def minHamming(x,y):#x,y are lists
    hamming = []
    for a in x:
        for b in y:
            a = np.array(a)
            b= np.array(b)
            #print(a)
            #print(b)
            vec = (a ^ b)
            #print (vec)
            hamming.append(sum (vec))  
    return min(hamming)/len(a)#, hamming
def aveHamming(x,y):#x,y are lists
    hamming = []
    for a in x:
        for b in y:
            a = np.array(a)
            b= np.array(b)
            #print(a)
            #print(b)
            vec = (a ^ b)
            #print (vec)
            hamming.append(sum (vec))
            AVE= sum(hamming)/len(hamming)
    return AVE/len(a)#, hamming

'''compute similarity of 2 agents'''
def simi(agent_1,agent_2):
    Num_inter =len(agent_1&agent_2)
    Num_uni = len(agent_1|agent_2)
    sim = (Num_inter/Num_uni)
    return sim 

'''
create all of the beliefs
'''
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

def deleteDuplicatedElementFromList(listx):
    resultList = []
    for item in listx:
        if not item in resultList:
            resultList.append(item)
    return resultList
'''modified version(with counting returning the percetage of each belief)'''
def deleteDuplicatedElementFromList2(listx):
    resultList = []
    counting=[]
    for item in listx:
        if not item in resultList:
            counting.append(listx.count(item)/len(listx))
            resultList.append(item)

    print(counting)
    return resultList,counting
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
        for j in range (i+1,an):
            Num_inter =len(agents[i]&agents[j])
            Num_uni = len(agents[i]|agents[j])
            similarity.append((Num_inter/Num_uni))
        simtotal.append(similarity)

    return similarity
''' one time for combine beliefs'''
def iterationHamm(agents,agent_number, iteration_times,threshold):

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
        if maxHamming(agents[index1],agents[index2])<=threshold:
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

    return averagesim, cardinality , agents

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()
#
def seriesplot(path,figurename,AVEsim, AVEsim_f,stdsim_f,AVEcard,index,AVEcard_f,stdcard_f,xaxis,countagt,belief_num):
    font = {'family' : 'serif',#'sans-serif':['Computer Modern Sans serif'],#Times New Roman',
            'weight' : 'light',
            #'size'   : list(figsize)[1]**1.6,
            }
    
    figsize = 6,4
    figure1, ax1 = plt.subplots(figsize=figsize)
    plt.plot(AVEsim)
    plt.ylabel('Similarity',font)
    plt.xlabel('Iterations',font)
    plt.title('Similarity-Iteration',font)
    labels =ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('serif') for label in labels]
    #plt.legend()
    plt.savefig(path+'Sim'+figurename+'.pdf')
    plt.show()
    
    figsize = 6,4
    figure2, ax2 = plt.subplots(figsize=figsize)
    plt.plot(AVEsim,color = 'brown')
    plt.errorbar(index, AVEsim_f, yerr = stdsim_f, fmt ='o',color = 'brown')
    plt.ylabel('Similarity',font)
    plt.xlabel('Iterations',font)
    plt.title('Similarity-Iteration with errorbar',font)
    labels =ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('serif') for label in labels]
    #plt.legend()
    plt.savefig(path+'SimErr'+figurename+'.pdf',dpi = 600)
    plt.show()

    figsize = 6,4
    figure3, ax3 = plt.subplots(figsize=figsize)
    #plt.figure(3)
    plt.plot(AVEcard)
    plt.ylabel('Cardinality',font)
    plt.xlabel('Iterations',font)
    plt.title('Cardinality-Iteration',font)
    labels =ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('serif') for label in labels]
    #plt.legend()
    plt.savefig(path+'Card'+figurename+'.pdf',dpi = 600)
    plt.show()
    
    figsize = 6,4
    figure4, ax4 = plt.subplots(figsize=figsize)
    #plt.figure(4)
    plt.plot(AVEcard,color = 'brown')
    plt.errorbar(index, AVEcard_f, yerr = stdcard_f, fmt ='o',color = 'brown')
    plt.ylabel('Cardinality',font)
    plt.xlabel('Iterations',font)
    plt.title('Cardinality-Iteration with errorbar',font)
    labels =ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('serif') for label in labels]
    #plt.legend()
    plt.savefig(path+'CardErr'+figurename+'.pdf',dpi = 600)
    plt.show()
    
    figsize = 6,4
    figure5, ax5 = plt.subplots(figsize=figsize)
    #plt.figure(5)
    plt.bar(xaxis,countagt,color = 'black',width = 0.4) 
    plt.xlabel('Agent Number',font)
    plt.ylabel('Times',font)
    plt.title("Times of covergence",font) 
    labels =ax5.get_xticklabels() + ax5.get_yticklabels()
    [label.set_fontname('serif') for label in labels]
    plt.savefig(path+'agt'+figurename+'.pdf',dpi = 600)
    plt.show()
    #print (belief_num)
    
    figsize = 6,4
    figure6, ax6 = plt.subplots(figsize=figsize)
    #plt.figure(6)
    plt.plot(belief_num,color = 'black') 
    plt.ylabel('Number of Final Beliefs',font)
    plt.xlabel('Iteration',font)
    plt.title("Number of Final Beliefs",font) 
    labels =ax6.get_xticklabels() + ax6.get_yticklabels()
    [label.set_fontname('serif') for label in labels]
    plt.savefig(path+'numbef'+figurename+'.pdf',dpi = 600)
    plt.show()



    
    