# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 01:57:25 2018

@author: liuzx
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def deleteDuplicatedElementFromList2(listx):
        resultList = []
        counting=[]
        for item in listx:
            if not item in resultList:
                counting.append(listx.count(item)/len(listx))
                resultList.append(item)
        counting.sort(reverse = True)        
        print(counting)
        return resultList,counting

def trans2dec(list_set_of_tuple):
    dec= []
    for index in range(len(list_set_of_tuple)):# range(len(list_set_of_tuple)):
#    print(set_of_tuple)
        for x in list_set_of_tuple[index]:
            a=0
            for i in range(len(x)):
                a = a+x[i]*math.pow(2,len(x)-i-1)
#            print(a)
            dec.append(a)
    
    return dec 
#def cal_mode(a):
#    counts = np.bincount(a)
#    return  np.argmax(counts)  
         
    
    
hui = [{(1,0,0,1),(0,1,1,0),(0,1,0,1),(0,1,1,1)},{(1,0,0,1),(0,1,0,0),(0,1,0,1),(0,1,1,1)},{(1,1,0,1),(0,1,0,0),(0,1,0,1),(0,1,1,1)},{(0,1,0,0),(1,1,0,1),(0,1,0,1),(0,1,1,1)}]
[rr,dd] = deleteDuplicatedElementFromList2(hui)
print(rr)
result = trans2dec(rr)
print (result)
#print (cal_mode(result))
#result = [1,2,1,4,5,5,5]
xstcick = np.arange(1, len(result)+1)
plt.bar(result,xstcick,color = 'black',width = 0.4) 
plt.title("histogram") 
plt.show()

