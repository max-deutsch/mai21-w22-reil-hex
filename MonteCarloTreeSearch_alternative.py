import hex_engine as hex
import math
import copy
from ModelBase import ModelBase
from RandomModel import RandomModel
import random

class Node:

    def __init__(self, parent):
        #self.action = action  # The action taken to end up in this state from the previous Node
        # self.state = board_state   # The current state of the board in this Node (Might be redundant with the action)
        # self.game_state: hex.hexPosition = game_state

        self.visitCount = 1  # is needed for n(s,a), init with 1 to avoid division by 0
        self.accumulatedValue = 0  # is needed for w(s,a)

        # todo persist action space instead of board?

        self.parent = parent
        self.children = {}  # key:action, value:Node

    def is_added_to_tree(self):
        # TODO visitCount > 1 maybe?
        return True


class MCTS:

    def __init__(self, model: ModelBase, c: float = math.sqrt(2)):
        self.c: float = c
        self.model: ModelBase = model

    def run(self, game_state: hex.hexPosition, num_iterations):
        root_node = Node(parent=None)

        for i in range(num_iterations):  # for many, many times (?) # todo: use time
            game_state_copy = copy.deepcopy(game_state)
            current_node = root_node
            while True:
                action = self.determine_action_by_uct(node=current_node, game_state=game_state_copy)
                if action is None:
                    break

                # take action, obtain reward (?), observe next state
                game_state_copy.board[action[0]][action[1]] = 1  # take action, always play as player 1
                current_node.visitCount += 1

                game_state_copy.board = game_state_copy.recodeBlackAsWhite()  # flipping the board to continue playing todo: when to flip?

                if action not in current_node.children:
                    next_node = Node(parent=current_node)  # adding current state to tree
                    # todo should visit count be increased?
                    current_node.children[action] = next_node
                    current_node = next_node
                    break
                current_node = current_node.children[action]

            # todo: is this correct that rewards are flipped, because the board was flipped after the last action?
            if game_state_copy.whiteWin():  # we always play as white
                reward = -1
            elif game_state_copy.blackWin():
                reward = 1
            elif action is None:
                reward = 0
            else:
                prior_probability = self.model.determine_prior_probability(game_state_copy.board)  # todo remember?
                state_value = self.model.determine_state_value(game_state_copy.board)
                reward = state_value

            # back propagate
            current_node.accumulatedValue += reward
            while current_node.parent is not None:
                current_node = current_node.parent
                # todo: Just guessing if the reward needs to be flipped here.
                current_node.accumulatedValue += reward * -1

        return root_node


    def determine_action_by_uct(self, node: Node, game_state: hex.hexPosition):
        max_value = float('-inf')
        action_max_value = None

        for action in game_state.getActionSpace():
            child: Node = node.children.get(action)
            cumulative_action_value = child.accumulatedValue if child else 0
            action_count = child.visitCount if child else 1  # 1 is default to avoid division by 0
            mean_action_value = cumulative_action_value / action_count  # aka exploitation term q(a', s)
            exploration_term = self.c * math.sqrt(math.log(node.visitCount)/action_count)

            value = mean_action_value + exploration_term
            if value > max_value:
                max_value = value
                action_max_value = action

        return action_max_value

    def is_final(self, node: Node):
        return False # todo

    # def take_action(self, node: Node, action):
    #
    #     next_game_state = copy.deepcopy(node.game_state)
    #     next_game_state.board[action[0], action[1]] = 1  # always play as player one
    #
    #
    #     # obtain reward? whiteWin?
    #     node.visitCount += 1
    #     # append to path?
    #
    #     child_node = Node(parent=current_node, game_state=next_game_state, action=action)  # is action needed?
    #     current_node.children[action] = child_node


if __name__ == "__main__":
    board_size = 2
    mcts = MCTS(model=RandomModel(board_size))
    game_state = hex.hexPosition(board_size)
    node = mcts.run(game_state=game_state, num_iterations=1000)
    pass