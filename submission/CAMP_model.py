import hex_engine as hex
import numpy as np
import torch
import copy
from random import choice

# from CNN
def evalCNN(CNN,game_state,device):
    board_array = []
    board_array.append(np.asarray(game_state.board))
    board_array = np.asarray(board_array)
    CNN.eval()  # needed when not training
    board_array = torch.from_numpy(board_array).unsqueeze(0).float().to(device)
    with torch.no_grad():
        determine_results = CNN(board_array)
    CNN.train()  # switches training back on
    return determine_results

def getActionCNN(CNN, game_state,device,board_size,exploit=True):
    game_state_empty = hex.hexPosition(board_size)
    full_action_space = game_state_empty.getActionSpace()
    determine_results = evalCNN(CNN, game_state, device)
    state_policy = determine_results['policy'].cpu()
    state_policy_probs = state_policy.detach().numpy()[0]
    #state_value = determine_results['value'].cpu()
    #reward = state_value.detach().numpy()[0]  # convert tensor to value
    #print(reward)
    #print(state_policy_probs)
    action_space = game_state.getActionSpace()
    # draw according to policy
    if exploit:  # exploit approach
        while True:
            action_i = np.argmax(state_policy_probs)
            action = full_action_space[action_i]
            if action in action_space:  # take next best if policy gives an action which is not possible
                return action
            state_policy_probs[action_i] = 0
    else:  # explore approach
        for i in range(100):
            action_i = np.random.choice(range(board_size * board_size), 1, p=state_policy_probs)[0]
            action = full_action_space[action_i]
            if action in action_space:  # redraw if policy gives an action which is not possible
                return action
    # if exploration does not draw anything within 100 tries, draw random
    return choice(action_space)

class hexCAMP():
    def __init__(self, model_path):

        self.size = 7

        if torch.cuda.is_available():
            self.CNN = torch.load(model_path).cpu()
        else:
            self.CNN = torch.load(model_path, map_location=torch.device('cpu'))

    def play(self, game, player, override=True):

        if override:
            for action in game.getActionSpace():
                game_copy = copy.deepcopy(game)
                game_copy.board[action[0]][action[1]] = player
                if game_copy.whiteWin() or game_copy.blackWin():
                    return action
                game_copy.board[action[0]][action[1]] = 2 if player == 1 else 1
                if game_copy.whiteWin() or game_copy.blackWin():
                    return action


        if player == 2: # TODO: this should be 1, but works better
            action = getActionCNN(self.CNN, game, "cpu", self.size, exploit=True)

        else:
            game.board = game.recodeBlackAsWhite(printBoard=False)
            action = getActionCNN(self.CNN, game, "cpu", self.size, exploit=True)
            game.board = game.recodeBlackAsWhite(printBoard=False)
            action = game.recodeCoordinates(action)

        return action
    




def modelVSmodel(board, player1, player2):
    while True:

        action = player1.play(game=board, player=1, override=False)
        board.board[action[0]][action[1]] = 1
        if board.whiteWin():
            break

        action = player2.play(game=board, player=2, override=False)
        board.board[action[0]][action[1]] = 2
        if board.blackWin():
            break

    # return 1 if model1 has won
    return board.whiteWin()


def modelVSrandom(board, player):
    while True:
        action = player.play(game=board, player=1, override=False)
        board.board[action[0]][action[1]] = 1
        if board.whiteWin():
            break

        board.playRandom(player=2)
        if board.blackWin():
            break

    return board.whiteWin()


def randomVSmodel(board, player):
    while True:
        board.playRandom(player=1)
        if board.whiteWin():
            break


        action = player.play(game=board, player=2, override=False)
        board.board[action[0]][action[1]] = 2
        if board.blackWin():
            break

    return board.blackWin()

if __name__ == "__main__":

    player1 = hexCAMP('champion.pt')
    player2 = hexCAMP('champion.pt')
    game = hex.hexPosition(7)
    human_player = 2
    game.humanVersusMachine(human_player,machine=lambda board: player1.play(board, 2 if human_player == 1 else 1))

    """
    myboard = hex.hexPosition(7)
    runs = 100
    white = 0
    black = 0
    for i in range(runs):
        if modelVSrandom(myboard, player1):
        #if modelVSmodel(myboard, player1, player2):  # True is exploit
            white += 1

        myboard.reset()
        if randomVSmodel(myboard, player1):
        #if not modelVSmodel(myboard, player2, player1):
            black += 1

        myboard.reset()
    print("Win rate as white: " + str(white / runs))
    print("Win rate as black: " + str(black / runs))
    print("Total win rate: " + str((white + black) / (2 * runs)))
    """