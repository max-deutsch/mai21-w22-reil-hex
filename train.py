import hex_engine as hex

import numpy as np
from CNN import CustomCNN, CustomDataset, trainCNN, getActionCNN
import torch
from torch import optim
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import subprocess

import config


# DO: exploit=False here for model evaluluation? yes!
def modelVSmodel(board, model1, model2):
    while True:

        action = getActionCNN(model1, board, "cpu", board.size, exploit=False)
        board.board[action[0]][action[1]] = 1
        if board.whiteWin():
            break

        board.board = board.recodeBlackAsWhite(printBoard=False)
        action = getActionCNN(model2, board, "cpu", board.size, exploit=False)
        board.board[action[0]][action[1]] = 1
        board.board = board.recodeBlackAsWhite(printBoard=False)
        if board.blackWin():
            break

    # return 1 if model1 has won
    return board.whiteWin()


def main():

    if not os.path.isdir("models"): os.makedirs("models")
    if not os.path.isdir("tmp_data"): os.makedirs("tmp_data")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # do not use GPU with multiprocessing

    if config.new_model:
        assert not os.path.isfile('models/champion.pt')
        CNN = CustomCNN(config.board_size).to(device)
        torch.save(CNN, 'models/champion.pt')
        iteration_history = []
        iteration_history.append(1.0)
    else:
        CNN = torch.load('models/champion.pt').to(device)
        iteration_history = np.loadtxt('models/iterations.txt').tolist()

    optimizer = optim.SGD(CNN.parameters(), lr=config.learning_rate, momentum=config.momentum)

    # count how many iterations the last champ lies in the past
    i_since_last_champ = 0
    for i in range(config.max_iterations):
        i_since_last_champ += 1
        print('Iterations since last champion: ' + str(i_since_last_champ))

        # clear game data
        for file in os.listdir('tmp_data/'):
            os.remove('tmp_data/' + file)
        game_time = time.time()

        subprocess.Popen(['sbatch', '-a', f'1-{config.num_parallel_workers}', 'run_worker.sh']).wait()

        time.sleep(3)


        while subprocess.check_output(['squeue', '-n', 'worker-hex-camp']).decode('utf-8').find('worker') != -1:
            time.sleep(1.0)



        mcts_boards = []
        mcts_values = []
        mcts_policies = []

        errors=0
        for parallel in range(config.num_parallel_workers):
            worker_i = parallel+1
            for i_game in range(config.num_worker_games):
                if os.path.isfile('tmp_data/' + str(worker_i) + "_" + str(i_game) + '_error.txt'):
                    errors+=1
                    print("Error in worker " + str(worker_i) + " , game " + str(i_game))
                else:
                    board = np.loadtxt('tmp_data/' + str(worker_i) + "_" + str(i_game) + '_board.txt')
                    board = board.reshape(board.shape[0], board.shape[1] // config.board_size, config.board_size)
                    value = np.loadtxt('tmp_data/' + str(worker_i) + "_" + str(i_game) + '_value.txt', dtype=float)
                    policy = np.loadtxt('tmp_data/' + str(worker_i) + "_" + str(i_game) + '_policy.txt', dtype=float)
                    mcts_boards += board.tolist()
                    mcts_values += value.tolist()
                    mcts_policies += policy.tolist()

        mcts_boards = np.asarray(mcts_boards, dtype=float)
        mcts_values = np.asarray(mcts_values, dtype=float)
        mcts_policies = np.asarray(mcts_policies, dtype=float)

        num_all_games = config.num_parallel_workers*config.num_worker_games
        print("Time for " + str(num_all_games) + " games: " + str(time.time() - game_time) + "s")
        print("--- thereof errors: " + str(errors))

        # learn for train_max_count
        train_time = time.time()
        for epoch in range(config.train_epochs):
            train_set = CustomDataset(mcts_boards, mcts_values, mcts_policies)
            loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=False)
            CNN.train()
            train_loss = trainCNN(CNN, loader, optimizer, device)
            #print("-Epoch " + str(i) + ": loss= " + str(train_loss))

        print("Time for " + str(config.train_epochs) + " epochs of training: " + str(time.time() - train_time) + "s")

        play_time = time.time()
        CNN_champ = torch.load('models/champion.pt').to(device)
        evalboard = hex.hexPosition(size=config.board_size)
        player1 = 0
        player2 = 0
        for j in range(config.eval_games):

            if modelVSmodel(evalboard, CNN, CNN_champ):
                player1 += 1

            evalboard.reset()

            if not modelVSmodel(evalboard, CNN_champ, CNN):
                player2 += 1

            evalboard.reset()

        white_wr = player1 / config.eval_games
        black_wr = player2 / config.eval_games
        tot_wr = (white_wr + black_wr) / 2
        print("Win rate as white: " + str(white_wr))
        print("Win rate as black: " + str(black_wr))
        print("Total win rate: " + str(tot_wr))
        print("Time for " + str(config.eval_games) + " games of evaluation: " + str(time.time() - play_time) + "s")

        if tot_wr > config.accept_wr:
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


if __name__ == "__main__":
    main()