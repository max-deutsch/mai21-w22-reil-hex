import hex_engine as hex
import math
import copy
import numpy as  np
from CNN import CustomCNN
from CNN import CustomDataset
from CNN import trainCNN
from CNN import evalCNN
from MonteCarloTreeSearch_alternative import MCTS
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import time
import multiprocessing as mp


#function to feed to mp.pool: run MCTS
# i can be used for tracking of day (or image) in the future
def mcts_to_pool(mcts,game_state,num_mcts_iterations,device,num_parallel_mcts):
    try:
        mcts_result = mcts.run(game_state=game_state, max_num_iterations=num_mcts_iterations, device=device, maxTime= 2) # 0.1/num_parallel_mcts)
        return game_state.board, mcts_result
    except:
        return None

    return None

#callback function to collect all results from async. mutliprocessing pool
def collect_mcts_results(result):
    global mcts_boards, mcts_values, mcts_policies
    if mcts_values is not None:
        board, mcts_result = result
        mcts_boards.append(np.asarray(board))
        mcts_values.append(mcts_result['value'])
        mcts_policies.append(mcts_result['policy'])


def main():
    global mcts_boards, mcts_values, mcts_policies

    board_size = 4
    num_mcts_iterations = 1000
    num_parallel_mcts = 16
    batch_size = num_parallel_mcts  # does not have to be

    game_state_empty = hex.hexPosition(board_size)
    full_action_space = game_state_empty.getActionSpace()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # do not use GPU with multiprocessing
    torch.set_num_threads(mp.cpu_count())
    CNN = CustomCNN(board_size).to(device)
    optimizer = optim.SGD(CNN.parameters(), lr=0.001, momentum=0.9)

    mcts = MCTS(model=CNN) # TODO: create new in each loop?

    # play until CNN converges
    # TODO: keep track previous models, losses, etc.
    for i in range(1):
        print("start new game")
        game_state = hex.hexPosition(board_size)

        # play a whole game until the end
        while True:
            # run 16 MCTS and feed into CNN
            mcts_boards = []
            mcts_values = []
            mcts_policies = []

            time_mcts = time.time()
            pool = mp.Pool(mp.cpu_count())  # create pools for parallelization
            for i in range(num_parallel_mcts):
                # non parallel version
                #try:
                    #mcts_result = mcts.run(game_state=game_state, max_num_iterations=num_mcts_iterations, device=device, maxTime= 2)# 0.1/num_parallel_mcts)
                    #mcts_boards.append(game_state.board)
                    #mcts_values.append(mcts_result['value'])
                    #mcts_policies.append(mcts_result['policy'])
                #except:  # ignore an mcts run if it throws an error
                #    continue
                # parallelized
                pool.apply_async(mcts_to_pool, args=(mcts,game_state,num_mcts_iterations,device,num_parallel_mcts), callback=collect_mcts_results)
            # join all processes of asynch pool totgether before starting new
            pool.close()
            pool.join()

            print("MCTS runs--- %s seconds ---" % (time.time() - time_mcts))

            train_time= time.time()
            mcts_boards = np.asarray(mcts_boards)
            mcts_values = np.asarray(mcts_values)
            mcts_policies = np.asarray(mcts_policies)

            train_set = CustomDataset(mcts_boards, mcts_values, mcts_policies)
            loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)  # Running on CPU
            CNN.train()
            trainCNN(CNN, loader, optimizer, device)
            print("train CNN--- %s seconds ---" % (time.time() - train_time))
            # Take "real" action
            print("take real action")
            determine_results = evalCNN(CNN,game_state,device)
            #state_value = determine_results['value']
            state_policy = determine_results['policy'].cpu()
            state_policy_probs = state_policy.detach().numpy()[0]
            action_space = game_state.getActionSpace()
            # draw according to policy

            ## probabilistic approach
            #while True:
            #    action_i = np.random.choice(range(board_size*board_size), 1, p=state_policy_probs)[0]
            #    action = full_action_space[action_i]
            #    if action in action_space: # redraw if policy gives an action which is not possible
            #        break

            ##exploit approach
            while True:
                action_i = np.argmax(state_policy_probs)
                action = full_action_space[action_i]
                if action in action_space: # take next best if policy gives an action which is not possible
                    break
                state_policy_probs[action_i]=0

            game_state.board[action[0]][action[1]] = 1  # take action, always play as player 1 (white)
            game_state.board = game_state.recodeBlackAsWhite(printBoard=True)
            if game_state.whiteWin() or game_state.blackWin():
                print("game ends wins")
                if game_state.whiteWin(): print("white wins")
                if game_state.blackWin(): print("black wins")
                break

    
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
    # #play against random player
    # myboard.humanVersusMachine()


if __name__ == "__main__":
    main()