import hex_engine as hex
import math
import copy
import numpy as np
from CNN import CustomCNN, CustomDataset, trainCNN, getActionCNN
from CNN import evalCNN
from MonteCarloTreeSearch import MCTS
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import time
import multiprocessing as mp
import os


# function to feed to mp.pool: run MCTS
def mcts_to_pool(mcts, game_state, num_mcts_iterations, device, maxTime):
    try:
        # Peter: 1s of maxTime is about 100 iterations
        num_iterations, mcts_result = mcts.run(
            game_state=game_state,
            max_num_iterations=num_mcts_iterations,
            device=device,
            max_seconds=maxTime)  # 0.1/num_parallel_mcts)
        return num_iterations, game_state.board, mcts_result
    except:
        return None

    return None


# callback function to collect all results from async. mutliprocessing pool
def collect_mcts_results(result):
    global mcts_boards, mcts_values, mcts_policies, mcts_iterations
    if mcts_values is not None:
        num_iterations, board, mcts_result = result
        mcts_iterations.append(num_iterations)  # figure out, how many iterations per time
        mcts_boards.append(np.asarray(board))
        mcts_values.append(mcts_result['value'])
        mcts_policies.append(mcts_result['policy'])


def game_to_pool(CNN, board_size, num_mcts_iterations, device, maxTime, mcts_c):
    try:
        game_state = hex.hexPosition(board_size)
        mcts = MCTS(model=CNN, c=mcts_c)  # TODO: create new in each loop?
        tmp_mcts_iterations = []
        tmp_mcts_boards = []
        tmp_mcts_values = []
        tmp_mcts_values_override = []
        tmp_mcts_policies = []
        # play a whole game until the end
        while True:

            num_iterations, mcts_result = mcts.run(game_state=game_state, max_num_iterations=num_mcts_iterations,
                                                   device=device, max_seconds=maxTime)  # 0.1/num_parallel_mcts)
            # TODO: sometimes append twice, sometimes not at all?
            tmp_mcts_boards.append(mcts_result['board'])
            tmp_mcts_values.append(mcts_result['value'])
            tmp_mcts_values_override.append(mcts_result['value'])
            tmp_mcts_policies.append(mcts_result['policy'])
            tmp_mcts_iterations.append(num_iterations)
            action = getActionCNN(CNN=CNN, game_state=game_state, device=device, board_size=board_size, exploit=False)
            game_state.board[action[0]][action[1]] = 1  # take action, always play as player 1 (white)
            game_state.board = game_state.recodeBlackAsWhite(printBoard=False)
            if game_state.whiteWin() or game_state.blackWin():
                break
        # override reward values
        reward = -1
        for i in reversed(range(len(tmp_mcts_values_override))):
            tmp_mcts_values_override[i] = reward
            reward *= -1
        # tmp_mcts_values_override can be returned instead of tmp_mcts_values
        #return tmp_mcts_iterations, tmp_mcts_boards, tmp_mcts_values, tmp_mcts_policies
        return tmp_mcts_iterations, tmp_mcts_boards, tmp_mcts_values_override, tmp_mcts_policies
    except:
        return None

    return None

def collect_game_results(result):
    global mcts_boards, mcts_values, mcts_policies, mcts_iterations, mcts_values2
    if result is not None:
        tmp_mcts_iterations, tmp_mcts_boards, tmp_mcts_values, tmp_mcts_policies = result
        mcts_boards += tmp_mcts_boards
        mcts_values += tmp_mcts_values
        mcts_policies += tmp_mcts_policies
        mcts_iterations += tmp_mcts_iterations


def modelVSrandom(board, model):
    while True:
        action = getActionCNN(model, board, "cpu", board.size, exploit=True)
        board.board[action[0]][action[1]] = 1
        if board.whiteWin():
            break

        board.playRandom(player=2)
        if board.blackWin():
            break

    return board.whiteWin()


def randomVSmodel(board, model):
    while True:
        board.playRandom(player=1)
        if board.whiteWin():
            break

        board.board = board.recodeBlackAsWhite(printBoard=False)
        action = getActionCNN(model, board, "cpu", board.size, exploit=True)
        board.board[action[0]][action[1]] = 1
        board.board = board.recodeBlackAsWhite(printBoard=False)
        if board.blackWin():
            break

    return board.blackWin()

def modelVSmodel(board, model1, model2):
    while True:

        action = getActionCNN(model1, board, "cpu", board.size, exploit=True)
        #action_ = getActionCNN(model2, board, "cpu", board.size, exploit=True)
        board.board[action[0]][action[1]] = 1
        #board.printBoard()
        if board.whiteWin():
            break

        board.board = board.recodeBlackAsWhite(printBoard=False)
        #action_ = getActionCNN(model1, board, "cpu", board.size, exploit=True)
        action = getActionCNN(model2, board, "cpu", board.size, exploit=True)
        board.board[action[0]][action[1]] = 1
        board.board = board.recodeBlackAsWhite(printBoard=False)
        #board.printBoard()
        if board.blackWin():
            break

    # return 1 if model1 has won
    return board.whiteWin()


def main():
    if not os.path.isdir("models"): os.makedirs("models")

    global mcts_boards, mcts_values, mcts_policies, mcts_iterations

    board_size = 4  # equals n x n board size

    num_parallel_games = 96
    batch_size = int(num_parallel_games / 4)  # does not have to be

    # MCTS parameter
    mcts_c = math.sqrt(2)
    max_mcts_time = 20
    num_mcts_iterations = 500

    # learning condition
    train_epochs = 10
    learning_rate = 0.01  # TODO: make schedule dependent. Decrease by factor after each few hundred steps
    momentum = 0.9


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # do not use GPU with multiprocessing
    torch.set_num_threads(mp.cpu_count())
    CNN = CustomCNN(board_size).to(device)
    #CNN = torch.load('models/model-1671493859.pt').to(device)
    optimizer = optim.SGD(CNN.parameters(), lr=learning_rate, momentum=momentum)

    #mcts = MCTS(model=CNN, c=mcts_c) # TODO: create new in each loop?


    CNN_current = copy.deepcopy(CNN)
    for i in range(1000):
        mcts_boards = []
        mcts_values = []
        mcts_policies = []
        mcts_iterations = []
        pool = mp.Pool(mp.cpu_count())
        game_time = time.time()
        for i in range(num_parallel_games):
            pool.apply_async(game_to_pool, args=(CNN, board_size, num_mcts_iterations, device, max_mcts_time, mcts_c),
                             callback=collect_game_results)
        pool.close()
        pool.join()
        print("Time for " + str(num_parallel_games) + " games: " + str(time.time() - game_time) + "s" )
        mcts_boards = np.asarray(mcts_boards)
        mcts_values = np.asarray(mcts_values)
        mcts_policies = np.asarray(mcts_policies)
        epochs = 0
        # learn for train_max_count
        train_time = time.time()
        for i in range(train_epochs):
            epochs += 1
            train_set = CustomDataset(mcts_boards, mcts_values, mcts_policies)
            loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
            CNN.train()
            train_loss = trainCNN(CNN, loader, optimizer, device)
            print("-Epoch " + str(i) + ": loss= " + str(train_loss))

        print("Time for " + str(epochs) + " epochs of training: " + str(time.time() - train_time) + "s")
        #print("Loss: " + str(train_loss))

        # save model after training
        ts = str(int(time.time()))
        model_name = 'model-' + ts + '.pt'
        torch.save(CNN, 'models/' + model_name)  # TODO: might be okay to do it this way
        CNN_new = copy.deepcopy(CNN)
        file1 = open("models/loss.txt", "a+")  # append mode
        file1.write(model_name + "    " + "loss: " + str(train_loss) + "\n")
        file1.close()


    # #Display the board in standard output
    # myboard.printBoard()
    # #Random playthrough
    # myboard.randomMatch(evaluate_when_full=False)
    # myboard.printBoard()
    # #check whether Black has won
    # myboard.blackWin(verbose=True)
    # #check whether White has won
    # myboard.whiteWin(verbose=True)
    # #print board with inverted colors
    # myboard.getInvertedBoard()
    # #get board as vector
    # myboard.getStateVector(inverted=False)
    # #reset the board
    # myboard.reset()

    # #play against CNN
    myboard = hex.hexPosition(size=board_size)
    #myboard.humanVersusMachine()

    # CNN vs random
    """
    CNN1 = torch.load('models/model-1671527638.pt').to(device)
    CNN2 = torch.load('models/4x4_3.pt').to(device)
    player1 = 0
    player2 = 0
    runs = 1000
    for i in range(runs):
        #if modelVSrandom(myboard, CNN1):
        if modelVSmodel(myboard, CNN1, CNN2):
            player1 += 1

        myboard.reset()
        #if randomVSmodel(myboard, CNN1):
        if modelVSmodel(myboard, CNN2, CNN1):
            player2 += 1

        myboard.reset()
    print("Win rate as white: " + str(player1 / runs))
    print("Win rate as black: " + str(player2 / runs))
    """

if __name__ == "__main__":
    main()