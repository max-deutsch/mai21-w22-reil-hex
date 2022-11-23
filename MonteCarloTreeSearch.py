# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:50:06 2022

@author: Christian
"""

from hex_engine import hexPosition
import math

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
        self.actionSpace #2d array of empty spots
        self.children = [] #nodes
        self.action = None # The action taken to end up in this node e.g. (1,2)
        self.visitCount = 0  # is needed for n(s,a)
        self.accumulatedValue = 0 # is needed for w(s,a)

        #self.state = None

        # self.value = 0


        # self.outcome # Only for leaf nodes. Storing the outcome of the game

        
    def isExpanded(self):
        return len(self.children) > 0
    
    def Expand(self):
        #insert all valid actions from that state into list of children
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
        root.children = initialState.getActionSpace()        
        for i in range(num_iterations):
            node = root
            
            #SELECTION
            while node.isExpanded():
                #select a child according to policy
                #slide 74
                list([ x.accumulatedValue/x.visitCount + math.sqrt(2)*() for x in node.children ])               
                
                
                # Determine a by UCT

                pass
                
            #invert Board?
            
            #EXPANSION
            #Get all possible actions for that leaf node and initialize them as children with visitcount 0
            newPossibleActions = node.state.getActionSpace()
            
            
            #SIMULATION
            
            #BACKUP