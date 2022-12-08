import hex_engine as hex
import math
import copy
import numpy as np
from CNN import CustomCNN
from CNN import CustomDataset
from CNN import trainCNN
from CNN import evalCNN
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import random
import time

class Node:

    def __init__(self, parent):
        self.visitCount = 1  # is needed for n(s,a), init with 1 to avoid division by 0
        self.accumulatedValue = 0  # is needed for w(s,a)
        self.parent = parent
        self.children = {}  # key:action, value:Node


class MCTS:

    def __init__(self, model, c: float = math.sqrt(2)):
        self.c: float = c
        self.model = model

    def run(self, game_state: hex.hexPosition, num_iterations):
        #run_start = time.time()
        root_node = Node(parent=None)

        for i in range(num_iterations):  # todo: use time

            game_state_copy = copy.deepcopy(game_state)
            current_node = root_node
            while True:
                action = self.determine_action_by_uct(node=current_node, game_state=game_state_copy)

                if action is None:
                    break

                # take action, obtain reward (?), observe next state
                game_state_copy.board[action[0]][action[1]] = 1  # take action, always play as player 1 (white)
                current_node.visitCount += 1
                # flipping the board to continue playing as player 1 (white)
                game_state_copy.board = game_state_copy.recodeBlackAsWhite()

                if action not in current_node.children:
                    next_node = Node(parent=current_node)  # adding current state to tree
                    # todo should visit count be increased?
                    current_node.children[action] = next_node
                    current_node = next_node
                    break
                current_node = current_node.children[action]

            # todo: is this correct that rewards are flipped, because the board was flipped after the last action?
            if game_state_copy.whiteWin():
                reward = -1
            elif game_state_copy.blackWin():
                reward = 1
            elif action is None:
                reward = 0
            else:
                determine_results = evalCNN(self.model,game_state_copy)
                state_value = determine_results['value']
                #prior_probability = determine_results['policy']
                reward = state_value.detach().numpy()[0]  # convert tensor to value


            # back propagate
            current_node.accumulatedValue += reward
            while current_node.parent is not None:
                current_node = current_node.parent
                current_node.accumulatedValue += reward * -1
        #print("MCTS run--- %s seconds ---" % (time.time() - run_start))
        return self.returnValues(root_node, game_state.size)

    def determine_action_by_uct(self, node: Node, game_state: hex.hexPosition):
        #max_value = float('-inf')
        #actions_max_value = []
        actions = []
        values = []

        for action in game_state.getActionSpace():
            child: Node = node.children.get(action)
            cumulative_action_value = child.accumulatedValue if child else 0
            action_count = child.visitCount if child else 1  # 1 is default to avoid division by 0
            mean_action_value = cumulative_action_value / action_count  # aka exploitation term q(a', s)
            exploration_term = self.c * math.sqrt(math.log(node.visitCount)/action_count)

            v = mean_action_value + exploration_term
            ## always take max
            # if v > max_value:
            #    actions_max_value = [action]
            #    max_value=v
            # elif v == max_value:
            #    actions_max_value.append(action)
            # choice = random.choice(actions_max_value)

            ##probabilistic
            if type(v) == float:
                value = v
            else:
                value = v[0]


            actions.append(action)
            values.append(value)

        if len(actions) == 0:
            return None

        values= np.array(values).astype(np.float)
        probs = np.exp(values) / np.sum(np.exp(values), axis=0)  # softmax of values
        action_i = np.random.choice(range(len(actions)), 1, p=probs)[0]

        return actions[action_i]

    def returnValues(self, node: Node, board_size):
        v = node.accumulatedValue / node.visitCount
        if type(v) == float:
            value = v
        else:
            value = v[0]
        game_state_empty = hex.hexPosition(board_size)
        full_action_space = game_state_empty.getActionSpace()

        pi_array = np.zeros(board_size * board_size)
        total_visits = 0
        for i, action in enumerate(full_action_space):

            child = node.children.get(action)

            if child is None:
                visits = 0
            else:
                visits = child.visitCount
                pi_array[i] = visits
            total_visits += visits

        pi_array = pi_array / total_visits

        return {'node': node,
                'value': value,
                'policy': pi_array}


if __name__ == "__main__":
    board_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CNN = CustomCNN(board_size).to(device)
    optimizer = optim.SGD(CNN.parameters(), lr=0.001, momentum=0.9)
    mcts = MCTS(model=CNN)

    game_state = hex.hexPosition(board_size)

    # run 16 MCTS and feed into CNN
    mcts_boards = []
    mcts_values = []
    mcts_policies = []

    for i in range(16): # TODO this loop could be parallelized
        mcts_result = mcts.run(game_state=game_state, num_iterations=3)
        mcts_boards.append(np.asarray(game_state.board))
        mcts_values.append(mcts_result['value'])
        mcts_policies.append(mcts_result['policy'])

    mcts_boards = np.asarray(mcts_boards)
    mcts_values = np.asarray(mcts_values)
    mcts_policies = np.asarray(mcts_policies)

    train_set = CustomDataset(mcts_boards, mcts_values, mcts_policies)
    loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=False)  # Running on CPU
    CNN.train()
    trainCNN(CNN, loader, optimizer, device)