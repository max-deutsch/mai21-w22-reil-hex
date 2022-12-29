new_model = True

board_size = 4  # equals n x n board size

max_iterations = 1000
num_parallel_workers = 12
num_worker_games = 8
batch_size = int(num_parallel_workers * num_worker_games / 4)  # does not have to be

# MCTS parameter
mcts_c = 1.41421356237
max_mcts_time = 20
num_mcts_iterations = 500

# learning condition
train_epochs = 10
learning_rate = 0.01  # TODO: make schedule dependent. Decrease by factor after each few hundred steps?
momentum = 0.9

# number of games to determine new champion + acceptance win rate
eval_games = 400
accept_wr = 0.55