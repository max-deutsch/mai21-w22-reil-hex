import hex_engine as hex
import math
import copy
import numpy as np

from CNN import evalCNN
import time

class Node:

    def __init__(self, parent):
        self.visitCount = 1  # is needed for n(s,a), init with 1 to avoid division by 0
        self.accumulatedValue = 0  # is needed for w(s,a)
        self.parent = parent
        self.children = {}  # key:action, value:Node
        self.endReward = None  # Denotes this Node as an end state with the respective reward

class MCTS:

    def __init__(self, model, c: float = math.sqrt(2)):
        self.c: float = c
        self.model = model

    def run(self, game_state: hex.hexPosition, max_num_iterations, device, max_seconds=1):
        root_node = Node(parent=None)
        end_time = time.time() + max_seconds
        for i in range(max_num_iterations):
            num_iterations = i
            self.loop(root_node, game_state, device)
            if max_seconds > 0 and time.time() >= end_time:
                break

        return num_iterations, self.returnValues(root_node, copy.deepcopy(game_state))


    def loop(self,root_node,game_state,device):
        
        game_state_copy = copy.deepcopy(game_state)

        current_node = root_node
        while True:
            if current_node.endReward:
                break

            action = self.determine_action_by_uct(node=current_node, game_state=game_state_copy)
            if action is None:
                break

            game_state_copy.board[action[0]][action[1]] = 1  # take action, always play as player 1 (white)
            current_node.visitCount += 1
            game_state_copy.board = game_state_copy.recodeBlackAsWhite()

            if action not in current_node.children:
                next_node = Node(parent=current_node)  # adding current state to tree
                current_node.children[action] = next_node
                current_node = next_node
                break
            current_node = current_node.children[action]

        if current_node.endReward:
            # using endReward instead of having to calculate winner again
            reward = current_node.endReward
            pass
        elif game_state_copy.blackWin():
            reward = 1
            current_node.endReward = reward
        elif action is None:
            reward = 0
        else:
            determine_results = evalCNN(CNN=self.model, game_state=game_state_copy,device=device)
            state_value = determine_results['value'].cpu()
            reward = state_value.detach().numpy()[0]  # convert tensor to value

        current_node.accumulatedValue += reward
        while current_node.parent is not None:
            current_node = current_node.parent
            reward = reward * -1
            current_node.accumulatedValue += reward
        return

    def determine_action_by_uct(self, node: Node, game_state: hex.hexPosition):
        actions = []
        values = []

        for action in game_state.getActionSpace():
            child: Node = node.children.get(action)
            cumulative_action_value = child.accumulatedValue if child else 0
            action_count = child.visitCount if child else 1  # 1 is default to avoid division by 0
            mean_action_value = cumulative_action_value / action_count  # aka exploitation term q(a', s)
            exploration_term = self.c * math.sqrt(math.log(node.visitCount)/action_count)

            v = mean_action_value + exploration_term
            if type(v) == float:
                value = v
            else:
                value = v[0]

            actions.append(action)
            values.append(value)

        if len(actions) == 0:
            return None

        values= np.array(values).astype(np.float)
        values = np.exp(values - np.max(values))
        probs = values / np.sum(values, axis=0)  # softmax of values
        action_i = np.random.choice(range(len(actions)), 1, p=probs)[0]

        return actions[action_i]

    def returnValues(self, node: Node, game_state):
        board_size = game_state.size
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
                'policy': pi_array,
                'board': game_state.board}
