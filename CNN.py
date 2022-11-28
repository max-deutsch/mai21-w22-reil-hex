##### Sources #####
# Repo of alpha zero approach with keras
# https://github.com/likeaj6/alphazero-hex

# Multi-Output CNN with pytorch on images
# https://medium.com/jdsc-tech-blog/multioutput-cnn-in-pytorch-c5f702d4915f

# Single output CNN from scratch
# https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/

# use arrays in NN with pytorch
# https://stackoverflow.com/questions/65017261/how-to-input-a-numpy-array-to-a-neural-network-in-pytorch
# https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797/3

#####         #####

# General libraries
import pandas as pd  # For working with dataframes
import numpy as np  # For working with image arrays
import cv2  # For transforming image
import matplotlib.pyplot as plt  # For representation
# For model building
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from skimage import io, transform
from torch.optim import lr_scheduler
from skimage.transform import AffineTransform, warp


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

        sample = {'board': board,
                  'value': value,
                  'policy': policy}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.board)


class CustomCNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_row):
        super(CustomCNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()

        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # value output
        self.conv_layer3a = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # self.batch_norm3a = nn.BatchNorm2d(32)
        self.relu3a = nn.ReLU()
        self.fc3a = nn.Linear(32, 1)

        # policy output
        self.conv_layer3b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # self.batch_norm3a = nn.BatchNorm2d(32)
        self.relu3b = nn.ReLU()
        self.fc3b = nn.Linear(32, num_row * num_row)
        self.softmax3b = nn.Softmax(dim=0)

        # self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        # self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Progresses data across layers
    def forward(self, x):
        common = self.conv_layer1(x)
        # common = self.batch_norm1(common)
        common = self.relu1(common)

        common = self.conv_layer2(common)
        # common = self.batch_norm2(common)
        common = self.relu2(common)

        # value output
        value = self.conv_layer3a(common)
        # value = self.batch_norm3a(value)
        value = self.relu3a(value)
        value = torch.flatten(value)
        value = self.fc3a(value)
        value = torch.tanh(value)

        # policy output
        policy = self.conv_layer3b(common)
        # policy = self.batch_norm3b(policy)
        policy = self.relu3b(policy)
        policy = torch.flatten(policy)
        policy = self.fc3b(policy)
        policy = self.softmax3b(policy)  # nn.LogSoftmax() might be more performant

        return value, policy


if __name__ == "__main__":

    np.random.seed(0)
    # create 10 random samples and append to lists
    board_array = []
    value_array = []
    policy_array = []

    for i in range(10):
        board_data = np.random.randint(low=-1, high=2, size=(7, 7))  # 7x7 board with values -1,0,1
        board_array.append(board_data)

        value_data = np.random.random_sample()
        value_array.append(value_data)

        policy_data = np.random.random_sample(5, )
        policy_array.append(policy_data)

    board_array = np.asarray(board_array)
    value_array = np.asarray(value_array)
    policy_array = np.asarray(policy_array)

    # dataset = MyDataset(numpy_data, numpy_target)
    # loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=False)  # Running on CPU

    # for i in range(len(board_array)):
    #    print(board_array[i])
    #    print(value_array[i])
    #    print(policy_array[i])

    # test DataLoader with custom dataset
    dataset = CustomDataset(board_array, value_array, policy_array)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=False)  # Running on CPU

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    CNN = CustomCNN(7)

    for batch_idx, sample in enumerate(loader):
        board = sample['board'].float().to(device)
        value, policy = CNN(board)
        print(value)
        print(policy)
        break
