##### Useful Sources #####
# Repo of alpha zero approach with keras
# https://github.com/likeaj6/alphazero-hex

# Multi-Output CNN with pytorch on images
# https://medium.com/jdsc-tech-blog/multioutput-cnn-in-pytorch-c5f702d4915f

# Single output CNN from scratch
# https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/

# use arrays in NN with pytorch
# https://stackoverflow.com/questions/65017261/how-to-input-a-numpy-array-to-a-neural-network-in-pytorch
# https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797/3

# alpha zero for othello
# https://github.com/2Bear/othello-zero/blob/910d9de816f33a8088b2a962f3816f6b679ecc0f/net.py#L64
#####         #####

# General libraries
import numpy as np  # For working with image arrays
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import hex_engine as hex
from random import choice
import time


class CustomDataset(Dataset):
    def __init__(self, board, value, policy, transform=None):
        self.board = torch.from_numpy(board)
        self.value = torch.from_numpy(value)
        self.policy = torch.from_numpy(policy)
        self.transform = transform

    def __getitem__(self, index):
        board = self.board[index]
        value = self.value[index]
        policy = self.policy[index]

        sample = {'board': board.unsqueeze(0).float(),
                  'value': value.float(),
                  'policy': policy.float()}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.board)


class CustomCNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_row):
        super(CustomCNN, self).__init__()
        use_channels = 64
        value_channels = 2
        policy_channels = 2

        # entry convolution layer
        self.conv_layer0 = nn.Conv2d(in_channels=1, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm0 = nn.BatchNorm2d(num_features=use_channels)
        self.relu0 = nn.ReLU()


        # residual block #1
        self.conv_layer1a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1a = nn.BatchNorm2d(num_features=use_channels)
        self.relu1a = nn.ReLU()
        self.conv_layer1b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1b = nn.BatchNorm2d(num_features=use_channels)
        self.relu1b = nn.ReLU()

        # residual block #2
        self.conv_layer2a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2a = nn.BatchNorm2d(num_features=use_channels)
        self.relu2a = nn.ReLU()
        self.conv_layer2b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2b = nn.BatchNorm2d(num_features=use_channels)
        self.relu2b = nn.ReLU()

        # residual block #3
        self.conv_layer3a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm3a = nn.BatchNorm2d(num_features=use_channels)
        self.relu3a = nn.ReLU()
        self.conv_layer3b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm3b = nn.BatchNorm2d(num_features=use_channels)
        self.relu3b = nn.ReLU()

        # residual block #4
        self.conv_layer4a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm4a = nn.BatchNorm2d(num_features=use_channels)
        self.relu4a = nn.ReLU()
        self.conv_layer4b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm4b = nn.BatchNorm2d(num_features=use_channels)
        self.relu4b = nn.ReLU()

        # residual block #5
        self.conv_layer5a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm5a = nn.BatchNorm2d(num_features=use_channels)
        self.relu5a = nn.ReLU()
        self.conv_layer5b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm5b = nn.BatchNorm2d(num_features=use_channels)
        self.relu5b = nn.ReLU()

        # residual block #6
        self.conv_layer6a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm6a = nn.BatchNorm2d(num_features=use_channels)
        self.relu6a = nn.ReLU()
        self.conv_layer6b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm6b = nn.BatchNorm2d(num_features=use_channels)
        self.relu6b = nn.ReLU()
        
        # residual block #7
        self.conv_layer7a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm7a = nn.BatchNorm2d(num_features=use_channels)
        self.relu7a = nn.ReLU()
        self.conv_layer7b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm7b = nn.BatchNorm2d(num_features=use_channels)
        self.relu7b = nn.ReLU()
        
        # residual block #8
        self.conv_layer8a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm8a = nn.BatchNorm2d(num_features=use_channels)
        self.relu8a = nn.ReLU()
        self.conv_layer8b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm8b = nn.BatchNorm2d(num_features=use_channels)
        self.relu8b = nn.ReLU()
        
        # residual block #9
        self.conv_layer9a = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm9a = nn.BatchNorm2d(num_features=use_channels)
        self.relu9a = nn.ReLU()
        self.conv_layer9b = nn.Conv2d(in_channels=use_channels, out_channels=use_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm9b = nn.BatchNorm2d(num_features=use_channels)
        self.relu9b = nn.ReLU()

        # value output
        self.conv_layerV1 = nn.Conv2d(in_channels=use_channels, out_channels=value_channels, kernel_size=1, stride=1, padding=0)
        self.batch_normV1 = nn.BatchNorm2d(value_channels)
        self.reluV1 = nn.ReLU()
        self.fcV1 = nn.Linear(value_channels * num_row * num_row, use_channels)
        self.reluV2 = nn.ReLU()
        self.fcV2 = nn.Linear(use_channels, 1)

        # policy output
        self.conv_layerP1 = nn.Conv2d(in_channels=use_channels, out_channels=policy_channels, kernel_size=1, stride=1, padding=0)
        self.batch_normP1 = nn.BatchNorm2d(policy_channels)
        self.reluP1 = nn.ReLU()
        self.fcP1 = nn.Linear(policy_channels * num_row * num_row, num_row * num_row)
        self.softmaxP1 = nn.Softmax(dim=1)

    # Progresses data across layers
    def forward(self, x):
        ## common CNN parts
        # entry convolution layer
        common = self.conv_layer0(x)
        common = self.batch_norm0(common)
        common = self.relu0(common)


        # residual block #1
        orig_common = common
        common = self.conv_layer1a(common)
        common = self.batch_norm1a(common)
        common = self.relu1a(common)
        common = self.conv_layer1b(common)
        common = self.batch_norm1b(common)
        common += orig_common
        common = self.relu1b(common)

        # residual block #2
        orig_common = common
        common = self.conv_layer2a(common)
        common = self.batch_norm2a(common)
        common = self.relu2a(common)
        common = self.conv_layer2b(common)
        common = self.batch_norm2b(common)
        common += orig_common
        common = self.relu2b(common)

        # residual block #3
        orig_common = common
        common = self.conv_layer3a(common)
        common = self.batch_norm3a(common)
        common = self.relu3a(common)
        common = self.conv_layer3b(common)
        common = self.batch_norm3b(common)
        common += orig_common
        common = self.relu3b(common)

        # residual block #4
        orig_common = common
        common = self.conv_layer4a(common)
        common = self.batch_norm4a(common)
        common = self.relu4a(common)
        common = self.conv_layer4b(common)
        common = self.batch_norm4b(common)
        common += orig_common
        common = self.relu4b(common)

        # residual block #5
        orig_common = common
        common = self.conv_layer5a(common)
        common = self.batch_norm5a(common)
        common = self.relu5a(common)
        common = self.conv_layer5b(common)
        common = self.batch_norm5b(common)
        common += orig_common
        common = self.relu5b(common)
        
        # residual block #6
        orig_common = common
        common = self.conv_layer6a(common)
        common = self.batch_norm6a(common)
        common = self.relu6a(common)
        common = self.conv_layer6b(common)
        common = self.batch_norm6b(common)
        common += orig_common
        common = self.relu6b(common)
        
        # residual block #7
        orig_common = common
        common = self.conv_layer7a(common)
        common = self.batch_norm7a(common)
        common = self.relu7a(common)
        common = self.conv_layer7b(common)
        common = self.batch_norm7b(common)
        common += orig_common
        common = self.relu7b(common)
        
        # residual block #8
        orig_common = common
        common = self.conv_layer8a(common)
        common = self.batch_norm8a(common)
        common = self.relu8a(common)
        common = self.conv_layer8b(common)
        common = self.batch_norm8b(common)
        common += orig_common
        common = self.relu8b(common)
        
        # residual block #9
        orig_common = common
        common = self.conv_layer9a(common)
        common = self.batch_norm9a(common)
        common = self.relu9a(common)
        common = self.conv_layer9b(common)
        common = self.batch_norm9b(common)
        common += orig_common
        common = self.relu9b(common)


        # value output
        value = self.conv_layerV1(common)
        value = self.batch_normV1(value)
        #value = self.reluV1(value)
        # value = torch.flatten(value)
        value = value.reshape(value.size(0), -1)
        value = self.fcV1(value)
        value = self.reluV2(value)
        value = value.reshape(value.size(0), -1)
        value = self.fcV2(value)
        value = torch.tanh(value)

        # policy output
        policy = self.conv_layerP1(common)
        policy = self.batch_normP1(policy)
        policy = self.reluP1(policy)
        # policy = torch.flatten(policy)
        policy = policy.reshape(policy.size(0), -1)
        policy = self.fcP1(policy)
        policy = self.softmaxP1(policy)  # nn.LogSoftmax() might be more performant

        return {'value': value, 'policy': policy}


def trainCNN(CNN, loader, optimizer, device):
    for epoch in range(1):  # TODO set epochs?
        train_loss = 0.0
        for batch_idx, sample_batched in enumerate(loader):
            # importing data and moving to GPU
            # print(sample_batched)
            board, value, policy = sample_batched['board'].to(device), sample_batched['value'].to(device), \
                                   sample_batched['policy'].to(device)

            output = CNN(board)
            value_est = output['value']
            policy_est = output['policy']
            loss1 = nn.MSELoss()(value_est, value.unsqueeze(1))
            loss2 = nn.CrossEntropyLoss()(policy_est, policy)
            # add regularizer?
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

    return train_loss.detach().numpy()


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


if __name__ == "__main__":

    num_rows = 4

    np.random.seed(0)
    # create 10 random samples and append to lists
    board_array = []
    value_array = []
    policy_array = []

    for i in range(50):
        board_data = np.random.randint(low=-1, high=2, size=(num_rows, num_rows))  # 7x7 board with values -1,0,1
        game_state = hex.hexPosition(num_rows)
        board_data = np.asarray(game_state.board)
        board_x3 = np.dstack((board_data, board_data, board_data))  # duplicate channels
        board_array.append(board_data)

        value_data = np.random.random_sample()
        value_array.append(value_data)

        policy_data = np.random.random_sample(num_rows * num_rows, )
        policy_array.append(policy_data)

    board_array = np.asarray(board_array)
    value_array = np.asarray(value_array)
    policy_array = np.asarray(policy_array)

    # test DataLoader with custom dataset
    dataset = CustomDataset(board_array, value_array, policy_array)
    loader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2, pin_memory=False)  # Running on CPU

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    CNN = CustomCNN(num_rows).to(device)
    optimizer = optim.SGD(CNN.parameters(), lr=0.001, momentum=0.9)

    trainCNN(CNN, loader, optimizer)

    # visualize what Con2v does
    #loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=False)  # Running on CPU
    #for batch_idx, sample_batched in enumerate(loader):
    #    board, value, policy = sample_batched['board'].float().to(device), sample_batched['value'].to(device), \
    #                           sample_batched['policy'].to(device)
    #    conv_board = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)(board)
    #    print(board)
    #    print(conv_board)
    #    break


