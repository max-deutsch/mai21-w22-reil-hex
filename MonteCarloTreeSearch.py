# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:50:06 2022

@author: Christian
"""

from hex_engine import hexPosition
import math
import copy

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

    def __init__(self,state,player):
        
        self.state                  #board state (array)
        self.actionSpace            #list of empty spots (x,y)
        
        self.visitCount = 0         # is needed for n(s,a)
        self.accumulatedValue = 0   # is needed for w(s,a)

        self.parent = None          #for backpropagation
        self.children = []          #nodes
        
        self.player = player        #who's turn is it

        
    def isExpanded(self):
        return len(self.children) > 0
    
    def Expand(self):
        #insert all valid actions from that state into list of children
        for action in self.actionSpace:
            newState = copy(self.state)
            newState[action[0]][action[1]] = 1 
            self.chilren.append(Node(newState))
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
            self.selection(root)
                
            #invert Board?
            
            #EXPANSION
            #Get all possible actions for that leaf node and initialize them as children with visitcount 0
            newPossibleActions = node.state.getActionSpace()
            
            
            #SIMULATION
            
            #BACKUP
        return
            
    def selection(self,root):
        
        node = root
        #SELECTION
        while node.isExpanded():
            #select a child according to policy
            #slide 74
            list([ x.accumulatedValue/x.visitCount + math.sqrt(2)*() for x in node.children ])               
            
            
            # Determine a by UCT

            pass
        return
        
    def expansion(self): 
        
        return