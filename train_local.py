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
import matplotlib.pyplot as plt


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
        mcts = MCTS(model=CNN, c=mcts_c)
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
        #reward = -1
        #for i in reversed(range(len(tmp_mcts_values_override))):
        #    tmp_mcts_values_override[i] = reward
        #    reward *= -1

        return tmp_mcts_iterations, tmp_mcts_boards, tmp_mcts_values, tmp_mcts_policies
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

# DO: exploit=False here for model evaluluation? yes!
def modelVSmodel(board, model1, model2, exploit=False):
    while True:

        action = getActionCNN(model1, board, "cpu", board.size, exploit=exploit)
        board.board[action[0]][action[1]] = 1
        if board.whiteWin():
            break

        board.board = board.recodeBlackAsWhite(printBoard=False)
        action = getActionCNN(model2, board, "cpu", board.size, exploit=exploit)
        board.board[action[0]][action[1]] = 1
        board.board = board.recodeBlackAsWhite(printBoard=False)
        if board.blackWin():
            break

    # return 1 if model1 has won
    return board.whiteWin()


def main():
    if not os.path.isdir("models"): os.makedirs("models")

    global mcts_boards, mcts_values, mcts_policies, mcts_iterations

    new_model = False

    board_size = 7  # equals n x n board size

    num_parallel_games = 1
    batch_size = int(num_parallel_games / 4)  # does not have to be

    # MCTS parameter
    mcts_c = math.sqrt(2)
    max_mcts_time = 20
    num_mcts_iterations = 1000

    # learning condition
    train_epochs = 10
    learning_rate = 0.01  # TODO: make schedule dependent. Decrease by factor after each few hundred steps?
    momentum = 0.9

    # number of games to determine new champion + acceptance win rate
    eval_games = 200
    accept_wr = 0.55


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # do not use GPU with multiprocessing
    torch.set_num_threads(mp.cpu_count())

    if new_model:
        assert not os.path.isfile('models/champion.pt')
        CNN = CustomCNN(board_size).to(device)
        torch.save(CNN, 'models/champion.pt')
        iteration_history = []
        iteration_history.append(1.0)
    else:
        CNN = torch.load('models/champion.pt').to(device)
        iteration_history = np.loadtxt('models/iterations.txt').tolist()

    optimizer = optim.SGD(CNN.parameters(), lr=learning_rate, momentum=momentum)

    # count how many iterations the last champ lies in the past
    i_since_last_champ = 0
    for i in range(0):
        i_since_last_champ += 1
        print('Iterations since last champion: ' + str(i_since_last_champ))
        mcts_boards = []
        mcts_values = []
        mcts_policies = []
        mcts_iterations = []
        pool = mp.Pool(mp.cpu_count())
        game_time = time.time()
        for parallel in range(num_parallel_games):
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
        for epoch in range(train_epochs):
            epochs += 1
            train_set = CustomDataset(mcts_boards, mcts_values, mcts_policies)
            loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
            CNN.train()
            train_loss = trainCNN(CNN, loader, optimizer, device)
            #print("-Epoch " + str(i) + ": loss= " + str(train_loss))

        print("Time for " + str(epochs) + " epochs of training: " + str(time.time() - train_time) + "s")


        CNN_champ = torch.load('models/champion.pt').to(device)
        evalboard = hex.hexPosition(size=board_size)
        player1 = 0
        player2 = 0
        for j in range(eval_games):

            if modelVSmodel(evalboard, CNN, CNN_champ):
                player1 += 1

            evalboard.reset()

            if not modelVSmodel(evalboard, CNN_champ, CNN):
                player2 += 1

            evalboard.reset()

        white_wr = player1 / eval_games
        black_wr = player2 / eval_games
        tot_wr = (white_wr + black_wr) / 2
        print("Win rate as white: " + str(white_wr))
        print("Win rate as black: " + str(black_wr))
        print("Total win rate: " + str(tot_wr))

        if tot_wr > accept_wr:
            # save old champion with timestamp when it was surpassed
            ts = str(int(time.time()))
            model_name = 'champ-' + ts + '.pt'
            torch.save(CNN_champ, 'models/' + model_name)
            # save new and better model as current champion
            torch.save(CNN, 'models/champion.pt')
            print('** New champion after ' + str(i_since_last_champ) + ' iterations **')
            iteration_history.append(i_since_last_champ)
            np.savetxt('models/iterations.txt', np.array(iteration_history), fmt='%d')
            i_since_last_champ = 0

            # plot inverse iteration graph
            plt.plot([x + 1 for x in range(len(iteration_history))], [1 / y for y in iteration_history], color="red", linestyle='dashed', marker='o')
            plt.xlabel('New champions')
            plt.ylabel('1 / iterations')
            plt.xticks(np.arange(1, len(iteration_history) + 1, step=1.))  # Set label locations.
            plt.savefig('models/plot.png')
            plt.close()



    # #play against CNN


    # CNN vs random
    ""
    myboard = hex.hexPosition(size=board_size)
    myboard.humanVersusMachine()
    exit()

    #"""
    CNN1 = torch.load('models_saved/champion.pt').to(device)
    CNN2 = torch.load('models_saved/champion4.pt').to(device)

    player1 = 0
    player2 = 0
    runs = 100
    for i in range(runs):
        #if modelVSrandom(myboard, CNN1):
        if modelVSmodel(myboard, CNN1, CNN2, False):  # True is exploit
            player1 += 1

        myboard.reset()
        #if randomVSmodel(myboard, CNN1):
        if not modelVSmodel(myboard, CNN2, CNN1, False):
            player2 += 1

        myboard.reset()
    print("Win rate as white: " + str(player1 / runs))
    print("Win rate as black: " + str(player2 / runs))
    print("Total win rate: " + str((player1+player2)/(2*runs)))
    #"""

if __name__ == "__main__":
    main()