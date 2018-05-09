# -*- coding: utf-8 -*-
"""
Created on Wed May  9 02:37:45 2018

@author: liuzx
"""

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()
    
def text_read(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]

    file.close()
    return content

'''Or we can use pickle'''
"""
import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('objs.pkl', 'w') as f:  # Python 2: 
    f=open(..., 'wb')
    pickle.dump([obj0, obj1, obj2], f)

# Getting back the objects:
with open('objs.pkl') as f:  # Python 2: 
    f=open(..., 'rb')
    obj0, obj1, obj2 = pickle.load(f)
"""