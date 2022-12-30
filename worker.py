import hex_engine as hex

import numpy as np
from CNN import getActionCNN

from MonteCarloTreeSearch import MCTS
import torch

import os

import config

if __name__ == "__main__":

    worker_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print('worker {} started'.format(worker_id))

    for i_game in range(config.num_worker_games):
        print('game {} started'.format(i_game))
        try:

            game_state = hex.hexPosition(config.board_size)
            device = torch.device("cpu")
            CNN = torch.load('models/current.pt').to(device)
            mcts = MCTS(model=CNN, c=config.mcts_c)
            tmp_mcts_iterations = []
            tmp_mcts_boards = []
            tmp_mcts_values = []
            # tmp_mcts_values_override = []
            tmp_mcts_policies = []
            # play a whole game until the end
            while True:

                num_iterations, mcts_result = mcts.run(game_state=game_state,
                                                       max_num_iterations=config.num_mcts_iterations,
                                                       device=device,
                                                       max_seconds=config.max_mcts_time)  # 0.1/num_parallel_mcts)
                # TODO: sometimes append twice, sometimes not at all?
                tmp_mcts_iterations.append(num_iterations)
                tmp_mcts_boards.append(mcts_result['board'])
                tmp_mcts_values.append(mcts_result['value'])
                # tmp_mcts_values_override.append(mcts_result['value'])
                tmp_mcts_policies.append(mcts_result['policy'])
                action = getActionCNN(CNN=CNN, game_state=game_state, device=device, board_size=config.board_size,
                                      exploit=False)
                game_state.board[action[0]][action[1]] = 1  # take action, always play as player 1 (white)
                game_state.board = game_state.recodeBlackAsWhite(printBoard=False)
                if game_state.whiteWin() or game_state.blackWin():
                    break
            # override reward values
            # reward = -1.0
            # for i in reversed(range(len(tmp_mcts_values_override))):
            #    tmp_mcts_values_override[i] = reward
            #    reward *= -1

            tmp_mcts_boards = np.asarray(tmp_mcts_boards)
            tmp_mcts_values = np.asarray(tmp_mcts_values)  # can be switched to normal
            tmp_mcts_policies = np.asarray(tmp_mcts_policies)

            # return tmp_mcts_boards, tmp_mcts_values, tmp_mcts_policies
            print(tmp_mcts_iterations)
            np.savetxt("tmp_data/" + str(worker_id) + "_" + str(i_game) + "_board.txt",
                       tmp_mcts_boards.reshape(tmp_mcts_boards.shape[0], -1))
            np.savetxt("tmp_data/" + str(worker_id) + "_" + str(i_game) + "_value.txt", tmp_mcts_values)
            np.savetxt("tmp_data/" + str(worker_id) + "_" + str(i_game) + "_policy.txt", tmp_mcts_policies)


        except Exception as e:
            print('game {} failed'.format(i_game))
            print(e)
            error = []
            error = np.asarray(error)
            np.savetxt("tmp_data/" + str(worker_id) + "_" + str(i_game) + "_error.txt", error)
