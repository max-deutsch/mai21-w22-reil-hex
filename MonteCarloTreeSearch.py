# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:50:06 2022

@author: Christian
"""

from hex_engine import hexPosition

class Node:
    
    def __init__(self):
        self.state = None
        self.visitCount = 0
        self.value = 0
        self.children ={}
        
    def isExpanded(self):
        return return len(self.children) > 0
    
    def Expand(self):
        #insert all valid actions fron that state into list of children
        
        
class MCTS:
    
    def __init__(self, model):
        #our CNN?
        self.model = model
    
    def run(self, initialState: hexPosition, num_iterations):
        
        #create tree with initial state
        root = Node()
        
        for range(num_iterations):
            node = root
            
            #SELECTION
            while node.isExpanded():
                #select a child and
                
            #invert Board?
            
            #EXPANSION
            
            #SIMULATION
            
            #BACKUP