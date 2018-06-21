# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:16:37 2018

@author: zl17868
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
import time
'''
create a belief world which includes all possible beliefs.
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

def deleteDuplicatedElementFromList2(listx):
    resultList = []
    counting=[]
    for item in listx:
        if not item in resultList:
            counting.append(listx.count(item)/len(listx))
            resultList.append(item)

    #print(counting)
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
def aveHamming(agents):#x,y are lists
    hamming = []
    AVEtotal = []
    an = len(agents)
    for i in range(an):
        for j in range (i+1,an):
            #average hamming distance of 2 beliefs:
            for a in agents[i]:
                for b in agents[j]:
                    a = np.array(a)
                    b= np.array(b)
                    #print(a)
                    #print(b)
                    vec = (a ^ b)
                    #print (vec)
                    hamming.append(sum (vec))
            AVE= sum(hamming)/len(hamming)
            AVEtotal.append(AVE/len(a))
    #AVEfinal = sum(AVEtotal)/len(AVEtotal)
    return AVEtotal#, hamming
'''Function for hamming distance'''
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
'''combine beliefs'''
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

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()
start1 = time.time()
#long running
#do something other
start = time.clock()
threshold = 0.9#(more than 0.1)
thre_store = []
sim_store = []
card_store = []
stdsim_store = []
stdcard_store = []
bn_store = []
counting_store=[]
avebtw_store=[]
std_b_num_store=[]
std_btween_store=[]
disbtween = []
ave_disbtween = []
srd_disbtween=[]
std_disbtween_store = []
disavebtw_store = []
while (threshold <=0.9):
    an = 100 #Number of agents
    sn = 5 #Number of propsitions
    N = 2000  # Times of iterations
    T = 50
    print (threshold)
    sim = []
    card = []
    countingstore=[]
    btween=[]
    AVEsim=[]
    AVEcard = []
    convergePos = []
    belief_num =[]
    #averagesim = [0]*T
    #cardinality = [0]*T
    #agents = initialise_agents(an, sn);
    for i in range (T):
        print(i)
        '''when change the initialise method,
        Remember to change the FILENAME and FIGURENAME'''
        #agents = initialise_agents(an, sn)
        agents = random_initialise(an, sn,create_world(sn))
        trans = copy.deepcopy(agents)
        #print (agents)
        (averagesim, cardinality, store) = iterationHamm(trans,an,N,threshold)
        print('asd')
        [store2,counting] = deleteDuplicatedElementFromList2(store)
        print('asdss')
        counting.sort()
        dec = trans2dec(store2)
       # print(simbetween)
        #print(store2)
        pos = copy.deepcopy(dec)
        #print (agents)        #print (pos)
        ''' to calculate average of simulations'''
        convergePos.append(pos)
        belief_num.append(len(store2))
        sim.append(averagesim)
        card.append(cardinality)
        countingstore.append(counting)
        #print (averagesim)
    sumsim = [0]*len(averagesim)
    sumcard = [0]*len(cardinality)
    sumcounting = [0]*len(counting)
    countagt = [0]*int(math.pow(2,sn))
    for i in range (T):
        sumsim = (np.sum([sumsim,sim[i]],axis = 0))
        sumcard =(np.sum([sumcard,card[i]],axis = 0))
        #sumcounting = (np.sum([sumcounting,countingstore[i]],axis = 0))
        countagt[int(convergePos[i][0])]=countagt[int(convergePos[i][0])]+1
    
    xaxis = np.arange(1, len(countagt)+1)
    Ave_b_num = sum(belief_num)/len(belief_num)
    AVEsim=sumsim/T
    AVEcard=sumcard/T
    #AVEcounting = sumcounting/T
    std_b_num =np.std(belief_num)
    stdsim =  np.std(sim,axis=0)
    stdcard =  np.std(card,axis=0)
    #stdconuting = np.std(countingstore,axis =0)
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
    counting_store.append(countingstore)
    thre_store.append(threshold)
    bn_store.append(Ave_b_num)
    stdsim_store.append(stdsim[-1])
    stdcard_store.append(stdcard[-1])
    sim_store.append(AVEsim[-1])
    card_store.append(AVEcard[-1])
    threshold = threshold +0.2
filename ='data_hammax_withbeliefnum50t100a'+str(T)
path = 'maxthreshold/'
text_save([thre_store, stdsim_store, stdcard_store,sim_store,card_store,bn_store,counting_store],path+filename,mode='a')

end1 = time.time()
print("Time1 used:",end1-start1)

elapsed = (time.clock() - start)
print("Time used:",elapsed)

'''
filename ='data'+str(an)+'_'+str(sn)+'_'+str(N)+'_'+str(T)+'_'+str(int(threshold*10))#+'single'
figurename = str(an)+'_'+str(sn)+'_'+str(N)+'_'+str(T)+'_'+str(int(threshold*10))#+'single'

path = 'figsSimlarity/'

plt.figure(1)
plt.plot(AVEsim)
plt.ylabel('Similarity')
plt.xlabel('Iterations')
#plt.ylim((0,1))
plt.title('Similarity-Iteration')
#plt.legend()
plt.savefig(path+'Sim'+figurename+'.png',dpi = 600)
plt.show()
plt.figure(2)
plt.plot(AVEsim,color = 'brown')
plt.errorbar(index, AVEsim_f, yerr = stdsim_f, fmt ='o',color = 'brown')
plt.ylabel('Similarity')
plt.xlabel('Iterations')
#plt.ylim((0,1))
plt.title('Similarity-Iteration with errorbar')
#plt.legend()
plt.savefig(path+'SimErr'+figurename+'.png',dpi = 600)
plt.show()
plt.figure(3)
plt.plot(AVEcard)
plt.ylabel('Cardinality')
plt.xlabel('Iterations')
#plt.ylim((0,int(math.pow(2,sn))))
plt.title('Cardinality-Iteration')
#plt.legend()
plt.savefig(path+'Card'+figurename+'.png',dpi = 600)
plt.show()
plt.figure(4)
plt.plot(AVEcard,color = 'brown')
plt.errorbar(index, AVEcard_f, yerr = stdcard_f, fmt ='o',color = 'brown')
plt.ylabel('Cardinality')
plt.xlabel('Iterations')
#plt.ylim((0,int(math.pow(2,sn))))
plt.title('Cardinality-Iteration with errorbar')
#plt.legend()
plt.savefig(path+'CardErr'+figurename+'.png',dpi = 600)
plt.show()
plt.figure(5)
plt.bar(xaxis,countagt,color = 'black',width = 0.4)
plt.xlabel('Agent Number')
plt.ylabel('Times')
plt.title("Times of covergence")
plt.savefig(path+'agt'+figurename+'.png',dpi = 600)
plt.show()
#print (belief_num)
plt.figure(6)
plt.plot(belief_num,color = 'black')
plt.ylabel('Number of Final Beliefs')
plt.xlabel('Iteration')
#plt.ylim((0,int(math.pow(2,sn))))
plt.title("Number of Final Beliefs")
plt.savefig(path+'numbef'+figurename+'.png',dpi = 600)
plt.show()
text_save([AVEsim, AVEcard, stdsim_f,stdcard_f,countagt,elapsed],path+filename+'txts',mode='a')
f= open(path+filename, 'wb')
pickle.dump([AVEsim, AVEcard, stdsim_f,stdcard_f,countagt,elapsed], f)
f.close()
'''