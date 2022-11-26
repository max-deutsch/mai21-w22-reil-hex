# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:50:06 2022

@author: Christian
"""

from hex_engine import hexPosition
from math import log,sqrt
from numpy import argmax
import random
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

    def __init__(self,boardState,actionSpace,player):
        
        self.state = boardState             #board state (array)
        self.actionSpace = actionSpace      #list of empty spots (x,y)
        
        self.visitCount = 1         # is needed for n(s,a), init with 1 to avoid division by 0
        self.accumulatedValue = 0   # is needed for w(s,a)

        self.parent = None          #for backpropagation
        self.children = []          #nodes
        
        self.player = player        #who's turn is it

        
    def isExpanded(self):
        return len(self.children) > 0
    
    def Expand(self):
        #insert one child for each valid action
        for action in self.actionSpace:
            newState = copy.deepcopy(self.state)
            #set action position to current player
            newState[action[0]][action[1]] = self.player 
            newActionSpace = copy.deepcopy(self.actionSpace)
            #remove the previous action from new action set
            newActionSpace.remove(action)
            self.children.append(Node(newState,newActionSpace, 2 if self.player == 1 else 1))
        pass

    
    #print nodes dfs style
    #TODO: bfs would be better, formatted somehow would be even better
    def printNode(self):
        print("Board State:")
        print(self.state)
        print("#Visited: " + str(self.visitCount-1))
        print("AccValue: " + str(self.accumulatedValue))
        print("Possible Actions:")
        print(self.actionSpace)
        print()
        for n in self.children:
            n.printNode()
        
class MCTS:
    
    def __init__(self, model):
        #our CNN?
        self.model = model
        self.root = None
        self.c = sqrt(2)
    
    def run(self, board: hexPosition, num_iterations):
        
        #create tree with initial state
        self.root = Node(board.board,board.getActionSpace(),1)
        
        for i in range(num_iterations):
            #SELECTION
            node = self.selection()

            #EXPANSION
            self.expansion(node)
                
            #invert Board? -> Expansion already inverts player
            #TODO check if or how board needs to be flipped aswell
            
            #SIMULATION
            #TODO give reward??
            reward = random.uniform(-1, 1)
            
            #BACKPROPAGATION
            #TODO check if back prop is correct
            node.accumulatedValue +=reward
            parent = node.parent
            while node.parent is not None:
                parent.accumulatedValue +=reward
                node = parent
                parent = node.parent
            
        return
            
    def selection(self):
        
        node = self.root
        node.visitCount+=1
        while node.isExpanded():
            #FIX LATER random policy for now
            # node = random.choice(node.children)
            
            # select a child according to policy - slide 74
            # Determine a by UCT 
            # TODO: check if formula is correct
            a = list([ x.accumulatedValue/x.visitCount + self.c*sqrt((log(node.visitCount)/x.visitCount)) for x in node.children ])
            # print(a)               
            best = argmax(a)
            node = node.children[best]
            node.visitCount+=1
        return node
        
    def expansion(self,node): 
        node.Expand()
        return
    def printTree(self):
        node = self.root
        node.printNode()        