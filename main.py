import hex_engine as hex
import math
import copy
import numpy as  np
from CNN import CustomCNN, CustomDataset, trainCNN, getActionCNN
from CNN import evalCNN
from MonteCarloTreeSearch import MCTS
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import time
import multiprocessing as mp
import os


#function to feed to mp.pool: run MCTS
# i can be used for tracking of day (or image) in the future
def mcts_to_pool(mcts,game_state,num_mcts_iterations,device, maxTime):
    try:
        # Peter: 1s of max_seconds is about 100 iterations
        num_iterations, mcts_result = mcts.run(
            game_state=game_state,
            max_num_iterations=num_mcts_iterations,
            device=device,
            max_seconds=maxTime)  # 0.1/num_parallel_mcts)
        return num_iterations, game_state.board, mcts_result
    except:
        return None

    return None

#callback function to collect all results from async. mutliprocessing pool
def collect_mcts_results(result):
    global mcts_boards, mcts_values, mcts_policies, mcts_iterations
    if mcts_values is not None:
        num_iterations, board, mcts_result = result
        mcts_iterations.append(num_iterations)  # figure out, how many iterations per time
        mcts_boards.append(np.asarray(board))
        mcts_values.append(mcts_result['value'])
        mcts_policies.append(mcts_result['policy'])

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
        board.board[action[0]][action[1]] = 1
        if board.whiteWin():
            break

        board.board = board.recodeBlackAsWhite(printBoard=False)
        action = getActionCNN(model2, board, "cpu", board.size, exploit=True)
        board.board[action[0]][action[1]] = 1
        board.board = board.recodeBlackAsWhite(printBoard=False)
        if board.blackWin():
            break

    # return 1 if model1 has won
    return board.whiteWin()

def main():
    if not os.path.isdir("models"): os.makedirs("models")

    global mcts_boards, mcts_values, mcts_policies, mcts_iterations

    board_size = 4  # equals 4x4 game field

    num_parallel_mcts = 48
    batch_size = int(num_parallel_mcts/2)  # does not have to be


    # MCTS parameter
    mcts_c = math.sqrt(2)
    max_mcts_time = 20
    num_mcts_iterations = 500

    # learning condition
    train_epochs = 10
    learning_rate = 0.01
    momentum = 0.9

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # do not use GPU with multiprocessing
    torch.set_num_threads(mp.cpu_count())
    CNN = CustomCNN(board_size).to(device)
    #CNN = torch.load('models/model-1670936299.pt').to(device)
    optimizer = optim.SGD(CNN.parameters(), lr=learning_rate, momentum=momentum)

    #mcts = MCTS(model=CNN, c=mcts_c) # TODO: create new in each loop?

    # play until CNN converges
    # TODO: keep track previous models, losses, etc.
    for i in range(1000):
        print("start new game")
        game_state = hex.hexPosition(board_size)
        losses = []

        # play a whole game until the end
        while True:
            # run multiple MCTS and feed into CNN
            mcts_boards = []
            mcts_values = []
            mcts_policies = []
            mcts_iterations = []

            time_mcts = time.time()
            pool = mp.Pool(mp.cpu_count())  # create pools for parallelization
            for i in range(num_parallel_mcts):
                # non parallel version
                #try:
                    #num_iterations, mcts_result = mcts.run(game_state=game_state, max_num_iterations=num_mcts_iterations, device=device, maxTime= 2)# 0.1/num_parallel_mcts)
                    #mcts_boards.append(game_state.board)
                    #mcts_values.append(mcts_result['value'])
                    #mcts_policies.append(mcts_result['policy'])
                    #mcts_iterations.append(num_iterations)
                #except:  # ignore an mcts run if it throws an error
                #    continue
                # parallelized
                mcts = MCTS(model=CNN, c=mcts_c)  # TODO: create new in each loop?
                pool.apply_async(mcts_to_pool, args=(mcts,game_state,num_mcts_iterations,device,max_mcts_time), callback=collect_mcts_results)
            # join all processes of asynch pool together before starting new
            pool.close()
            pool.join()
            #print(mcts_iterations)
            print("MCTS runs--- %s seconds ---" % (time.time() - time_mcts))
            mcts_boards = np.asarray(mcts_boards)
            mcts_values = np.asarray(mcts_values)
            mcts_policies = np.asarray(mcts_policies)
            train_time = time.time()
            for i in range(train_epochs):
                train_set = CustomDataset(mcts_boards, mcts_values, mcts_policies)
                loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
                CNN.train()
                loss = trainCNN(CNN, loader, optimizer, device)
                #loss=loss.detach().numpy()
                #losses.append(loss)
            print("train CNN--- %s seconds ---" % (time.time() - train_time))
            # Take "real" action
            #print("take real action")
            action = getActionCNN(CNN=CNN,game_state=game_state,device=device,board_size=board_size, exploit=False)
            game_state.board[action[0]][action[1]] = 1  # take action, always play as player 1 (white)
            game_state.board = game_state.recodeBlackAsWhite(printBoard=False)
            if game_state.whiteWin() or game_state.blackWin():
                print("game ends wins")
                if game_state.whiteWin(): print("white wins")
                if game_state.blackWin(): print("black wins")
                break

        # save model after each full game
        ts = str(int(time.time()))
        model_name = 'model-' + ts + '.pt'
        torch.save(CNN, 'models/' + model_name)  # TODO: might be okay to do it this way
        #file1 = open("models/loss.txt", "a+")  # append mode
        #file1.write(model_name + "    " + "avg.loss: " + str(np.mean(losses)) + "\n")
        #file1.close()


    
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
    CNN = torch.load('models/4x4_2.pt').to(device)
    #CNN1 = CNN
    #CNN2 = CNN
    player1 = 0
    player2 = 0
    runs = 1000
    for i in range(runs):
        if modelVSrandom(myboard, CNN):
            player1 += 1

        myboard.reset()
        if randomVSmodel(myboard, CNN):
            player2 += 1

        myboard.reset()
    print("Win rate as white: " + str(player1 / runs))
    print("Win rate as black: " + str(player2 / runs))
    """

if __name__ == "__main__":
    main()