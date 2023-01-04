new_model = False

board_size = 7 # equals n x n board size

max_iterations = 10000
num_parallel_workers = 128
num_worker_games = 2
batch_size = 32 # int(num_parallel_workers * num_worker_games / 16)  # does not have to be

# MCTS parameter
mcts_c = 1.41421356237
max_mcts_time = 20
num_mcts_iterations = 1000

# learning condition
train_epochs = 10
learning_rate = 0.001  # TODO: make schedule dependent. Decrease by factor after each few hundred steps?
momentum = 0.9

# number of games to determine new champion + acceptance win rate
eval_games = 100
accept_wr = 0.6
