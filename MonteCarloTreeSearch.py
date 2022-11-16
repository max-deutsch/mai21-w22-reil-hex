# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:50:06 2022

@author: Christian
"""

from hex_engine import hexPosition

class Node:

    # [
    #   [0, 0]
    #   [0, 0]
    # ]

    # n(s,a)
    # Visit count per child.

    # state s [0, 0, 0, 1]
    # actions [(0,0), ..]



    # TODO: invalid actions filtering

    def __init__(self):
        self.parent = None
        self.children = {}
        self.action = None # The action taken to end up in this node e.g. (1,2)
        self.visitCount = 0  # is needed for n(s,a)
        self.accumulatedValue = 0 # is needed for w(s,a)

        # self.state = None

        # self.value = 0


        # self.outcome # Only for leaf nodes. Storing the outcome of the game

        
    def isExpanded(self):
        return len(self.children) > 0
    
    def Expand(self):
        #insert all valid actions fron that state into list of children
        pass

    def determineActionWithUCT(self):
        # for
        return
        
class MCTS:
    
    def __init__(self, model):
        #our CNN?
        self.model = model
    
    def run(self, initialState: hexPosition, num_iterations):
        
        #create tree with initial state
        root = Node()
        
        for i in range(num_iterations):
            node = root
            
            #SELECTION
            while node.isExpanded():
                #select a child and

                # Determine a by UCT

                pass
                
            #invert Board?
            
            #EXPANSION
            
            #SIMULATION
            
            #BACKUP