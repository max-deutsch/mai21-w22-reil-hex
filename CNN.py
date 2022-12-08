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
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2)
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()

        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # value output
        self.conv_layer3a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.batch_norm3a = nn.BatchNorm2d(32)
        self.relu3a = nn.ReLU()
        self.fc3a = nn.Linear(32, 1)

        # policy output
        self.conv_layer3b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.batch_norm3b = nn.BatchNorm2d(32)
        self.relu3b = nn.ReLU()
        self.fc3b = nn.Linear(32, num_row * num_row)
        self.softmax3b = nn.Softmax(dim=1)

    # Progresses data across layers
    def forward(self, x):
        #print(x)
        # common CNN parts
        common = self.conv_layer1(x)
        common = self.batch_norm1(common)
        common = self.relu1(common)

        common = self.conv_layer2(common)
        common = self.batch_norm2(common)
        common = self.relu2(common)

        # value output
        value = self.conv_layer3a(common)
        value = self.batch_norm3a(value)
        value = self.relu3a(value)
        # value = torch.flatten(value)
        value = value.reshape(value.size(0), -1)
        value = self.fc3a(value)
        value = torch.tanh(value)

        # policy output
        policy = self.conv_layer3b(common)
        policy = self.batch_norm3b(policy)
        policy = self.relu3b(policy)
        # policy = torch.flatten(policy)
        policy = policy.reshape(policy.size(0), -1)
        policy = self.fc3b(policy)
        policy = self.softmax3b(policy)  # nn.LogSoftmax() might be more performant

        return {'value': value, 'policy': policy}


def trainCNN(CNN, loader, optimizer, device):
    for epoch in range(1):  # TODO set epochs?
        for batch_idx, sample_batched in enumerate(loader):
            # importing data and moving to GPU
            # print(sample_batched)
            board, value, policy = sample_batched['board'].to(device), sample_batched['value'].to(device), \
                                   sample_batched['policy'].to(device)
            # print(board.size())

            # zero the parameter gradients

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
        #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 10, loss.item()))
    ts = str(int(time.time()))
    torch.save(CNN, 'models/model-' + ts + '.pt')

def evalCNN(CNN,game_state):
    board_array = []
    board_array.append(np.asarray(game_state.board))
    board_array = np.asarray(board_array)
    CNN.eval()  # needed when not training
    board_array = torch.from_numpy(board_array).unsqueeze(0).float()
    determine_results = CNN(board_array)
    CNN.train()  # switches training back on
    return determine_results

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


