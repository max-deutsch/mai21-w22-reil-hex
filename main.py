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
def mcts_to_pool(mcts,game_state,num_mcts_iterations,device,num_parallel_mcts):
    try:
        # Peter: 1s of maxTime is about 100 iterations
        num_iterations, mcts_result = mcts.run(game_state=game_state, max_num_iterations=num_mcts_iterations, device=device, maxTime= 3) # 0.1/num_parallel_mcts)
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


def main():
    if not os.path.isdir("models"): os.makedirs("models")

    global mcts_boards, mcts_values, mcts_policies, mcts_iterations

    board_size = 4  # equals 4x4 game field
    num_mcts_iterations = 1000
    num_parallel_mcts = 16
    batch_size = num_parallel_mcts  # does not have to be

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # do not use GPU with multiprocessing
    torch.set_num_threads(mp.cpu_count())
    CNN = CustomCNN(board_size).to(device)
    #CNN = torch.load('models/model-1670936299.pt').to(device)
    optimizer = optim.SGD(CNN.parameters(), lr=0.01, momentum=0.9)

    mcts = MCTS(model=CNN) # TODO: create new in each loop?

    # play until CNN converges
    # TODO: keep track previous models, losses, etc.
    for i in range(1):
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
                #mcts = MCTS(model=CNN)  # TODO: create new in each loop?
                pool.apply_async(mcts_to_pool, args=(mcts,game_state,num_mcts_iterations,device,num_parallel_mcts), callback=collect_mcts_results)
            # join all processes of asynch pool together before starting new
            pool.close()
            pool.join()
            #print(mcts_iterations)
            print("MCTS runs--- %s seconds ---" % (time.time() - time_mcts))
            train_time= time.time()
            mcts_boards = np.asarray(mcts_boards)
            mcts_values = np.asarray(mcts_values)
            mcts_policies = np.asarray(mcts_policies)
            #print(mcts_iterations)
            train_set = CustomDataset(mcts_boards, mcts_values, mcts_policies)
            loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
            CNN.train()
            loss = trainCNN(CNN, loader, optimizer, device)
            loss=loss.detach().numpy()
            losses.append(loss)
            print("train CNN--- %s seconds ---" % (time.time() - train_time))
            # Take "real" action
            #print("take real action")
            action = getActionCNN(CNN=CNN,game_state=game_state,device=device,board_size=board_size, exploit=True)
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
        file1 = open("models/loss.txt", "a+")  # append mode
        file1.write(model_name + "    " + "avg.loss: " + str(np.mean(losses)) + "\n")
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
    #myboard = hex.hexPosition(size=board_size)
    #myboard.humanVersusMachine()


if __name__ == "__main__":
    main()