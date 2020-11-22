import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import numpy as np

import matplotlib.pyplot as plt

import logging


logger = logging.getLogger(__name__)
LOG_INTERVAL = 25


class Net(nn.Module):
    def __init__(self, input_features, output_dim):
        super(Net, self).__init__()
        # 1-dimensional convolutional layer
        self.conv0 = nn.Conv1d(input_features, 128, output_dim, stride=1, padding=0)
        self.conv1 = nn.Conv1d(128, 128, output_dim, stride=1, padding=2)

        # max pooling layer
        self.pool1 = nn.MaxPool1d(5, 2)

        # fully-connected layer
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, output_dim)

        # softmax output
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        inp = x.view(32, -1, 187)
        C = self.conv0(inp)

        # first conv layer
        C11 = self.conv0(inp)
        A11 = F.relu(C11)
        C12 = self.conv1(A11)
        S11 = torch.add(C12, C)
        M11 = self.pool1(S11)

        # second conv layer
        C21 = self.conv1(M11)
        A21 = F.relu(C21)
        C22 = self.conv1(A21)
        S21 = torch.add(C22, M11)
        M21 = self.pool1(S21)

        # third conv layer
        C31 = self.conv1(M21)
        A31 = F.relu(C31)
        C32 = self.conv1(A31)
        S31 = torch.add(C32, M21)
        M31 = self.pool1(S31)

        # fourth conv layer
        C41 = self.conv1(M31)
        A41 = F.relu(C41)
        C42 = self.conv1(A41)
        S41 = torch.add(C42, M31)
        M41 = self.pool1(S41)

        # last layer
        C51 = self.conv1(M41)
        A51 = F.relu(C51)
        C52 = self.conv1(A51)
        S51 = torch.add(C52, M41)
        M51 = self.pool1(S51)

        # flatten the output of the last layer
        F1 = M51.view(32, -1)

        D1 = self.fc1(F1)
        A6 = F.relu(D1)
        D2 = self.fc2(A6)

        return self.softmax(D2)


def test(model, dataLoader, train_on_gpu):
    # Specify the heartbeat classes from above
    classes = {
        0: 'N - Normal Beat',
        3: 'S - Supraventricular premature or ectopic beat',
        2: 'V - Premature ventricular contraction',
        4: 'F - Fusion of ventricular and normal beat',
        1: 'Q - Unclassified beat'}

    model.eval()

    for data, _ in dataLoader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data = data.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data.float())
        # calculate the batch loss
        # update test loss
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        most_frequent_val = np.bincount(pred).argmax()
        print("Your beat is " + str(classes[most_frequent_val]))


### 기본, 모델 클래스 선언 필요
model = Net(input_features=1, output_dim=5)
model.load_state_dict(torch.load("./Model/mnist_cnn.pt"))
mitbih_test_loc = "C:/Users/Lee/Downloads/archive/mitbih_train.csv"
ind_data = "./data/temp_data.csv"


def test_with_predata(df, data=True):
    if not data:
        test_df = pd.read_csv(mitbih_test_loc, header=None)
        test_df = test_df.astype(float)
        c = test_df.groupby(187, group_keys=False).apply(lambda test_df : test_df.sample(1))
        d = test_df.groupby(187, group_keys=False)

        # correct this one
        # N, Q, V, S, F (Tested one)
        # N, S, V, F, Q (Trained one)
        target = 0
        pre_temp_data = c.iloc[target, :187]
        pre_temp_label = c.iloc[target, :188]
        print(pre_temp_data)
    else:
        pre_temp_data = df

    plt.plot(pre_temp_data)
    plt.show()

    real_data = np.zeros((32, 187))
    for i in range(32):
        real_data[i] = pre_temp_data

    junk_label = np.zeros(32, )

    test_dataset = D.TensorDataset(torch.tensor(real_data),
                                   torch.tensor(junk_label))

    test_loader = D.DataLoader(test_dataset, batch_size=32,
                               num_workers=0, drop_last=True)

    # evaluate(model, temp_data)
    test(model, test_loader, False)


def test_with_realdata():
    def translate(value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return round(rightMin + (valueScaled * rightSpan), 3)

    df = np.loadtxt(ind_data, dtype='str', delimiter = ",")[0]
    df = df.astype(int)
    df = df / 100
    shape = df.shape[0]
    print("First value is:", df[0], "Length is:", shape)

    max_val = np.max(df)
    min_val = np.min(df)

    lambda_func = lambda t: translate(t, min_val, max_val, 0, 1)
    df = np.array([lambda_func(elements) for elements in df])

    print(df)

    if shape > 187:
        while df.shape[0] > 187:
            column = df.shape[0]
            remain = 20

            for i in list(range(0, column)):
                try:
                    if i % remain == 0:
                            df = df.drop(df.columns[i], axis=1)
                            if df.shape[0] == 187:
                                break
                except IndexError:
                    break
    elif shape < 187:
        idx = 1

        while df.shape[0] < 187:
            column = df.shape[0]
            remain = 2

            for i in list(range(0, (column - 1)*2)):
                if i % remain == 0 and i != 0:
                    value = ((df[i-1] + df[i]) / 2)
                    df = np.insert(df, i, value, axis=0)
                    idx += 1

                    # print(df)
                    if df.shape[0] == 187:
                        break

    print(df.shape)
    test_with_predata(df)


if __name__ == "__main__":
    # test_with_predata()
    test_with_realdata()