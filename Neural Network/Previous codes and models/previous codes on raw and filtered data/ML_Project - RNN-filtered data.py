import pandas as pd
import numpy as np
import glob
import scipy.signal
import scipy
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch import optim


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=306, hidden_size=500, batch_first=True, bias=False)
        #self.h0 = torch.randn(1, 128, 40)  # 40-> the number of hidden layer features
        self.FC = torch.nn.Linear(500, 2)  # fully connected layer
        #self.bn = nn.BatchNorm1d(num_features=306)

    def forward(self, input):
        input = input*10000000000000000
        #input = self.bn(input)
        h0 = torch.randn(1, input.shape[0], 500)  # 40-> the number of hidden layer features
        h0 = h0.to(dev)
        output, hn = self.rnn(input, h0)
        hn = hn.view(-1, 500)
        hn = self.FC(hn)
        return hn



if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True

    # X
    pathX = r'..\decoding-the-human-brain\csvs\train\X'  # use your path
    all_filesX = glob.glob(pathX + "/*.csv")
    print("all train files")
    print(all_filesX)
    path_to_test_dataX = r'..\decoding-the-human-brain\csvs\test\X'  # use your path
    all_test_filesX = glob.glob(path_to_test_dataX + "/*.csv")
    print("all test files")
    print(all_test_filesX)

    dfX = pd.concat((pd.read_csv(f) for f in all_filesX))  # to read only 5 rows you can write f, nrows=5
    print(dfX.iloc[0])
    df_testX = pd.concat((pd.read_csv(f_test) for f_test in all_test_filesX))
    #print(df_test[["y "]])  # this worked
    X = dfX
    print("x shpe:", X.shape)
    print("x [0.0]:", X.values[0, 0])
    X_test = df_testX

    # Y
    pathY = r'..\decoding-the-human-brain\csvs\train\Y'  # use your path
    all_filesY = glob.glob(pathY + "/*.csv")
    print("all train files")
    print(all_filesY)
    path_to_test_dataY = r'..\decoding-the-human-brain\csvs\test\Y'  # use your path
    all_test_filesY = glob.glob(path_to_test_dataY + "/*.csv")
    print("all test files")
    print(all_test_filesY)

    dfY = pd.concat((pd.read_csv(f) for f in all_filesY))  # to read only 5 rows you can write f, nrows=5
    print(dfY.iloc[0])
    df_testY = pd.concat((pd.read_csv(f_test) for f_test in all_test_filesY))
    # print(df_test[["y "]])  # this worked
    Y = dfY
    print("y shpe:", Y.shape)
    print("y [0.0]:", Y.values[0, 0])
    print("y", Y)
    Y_test = df_testY




    # creating tensor from DataFrame dataset
    X = torch.tensor(X.values)
    Y = torch.tensor(Y.values).view(-1)
    X_test = torch.tensor(X_test.values)
    Y_test = torch.tensor(Y_test.values).view(-1)


    # create training and validation split
    split_size = int(0.8 * len(dfX))
    index_list = list(range(len(dfX)))
    train_idx, valid_idx = index_list[:split_size], index_list[split_size:]

    # create iterator objects for train and valid datasets
    x_tr = torch.tensor(X[:split_size], dtype=torch.float)
    y_tr = torch.tensor(Y[:split_size], dtype=torch.long)
    train = TensorDataset(x_tr, y_tr)
    trainloader = DataLoader(train, batch_size=64, shuffle=False)

    x_val = torch.tensor(X[split_size:], dtype=torch.float)
    y_val = torch.tensor(Y[split_size:], dtype=torch.long)
    valid = TensorDataset(x_val, y_val)
    validloader = DataLoader(valid, batch_size=64, shuffle=False)

    model = Model()
    model = model.to(dev)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)

    for epoch in range(0, 30):  # run the model for 10 epochs
        train_loss, valid_loss = [], []
        training_corrects, valid_corrects = 0.0, 0.0
        training_incorrects, valid_incorrects = 0.0, 0.0
        print("epoch : ", epoch)
        # training part, 0.0
        model.train()
        for data, target in trainloader:

            data = data.to(dev)
            target = target.to(dev)


            optimizer.zero_grad()

            # 1. forward propagation
            data = data.view(-1, 306, 250).permute(0, 2, 1)  # (batch, seq, feature)

            output = model(data)

            #print(" training output:", output)
            _, preds = torch.max(output.data, 1)

            # 2. loss calculation
            loss = loss_function(output, target)

            # 3. backward propagation
            loss.backward()
            # 4. weight optimization
            optimizer.step()
            train_loss.append(loss.item())

            # statistics
            training_corrects += torch.sum(preds == target.data)
            training_incorrects += torch.sum(preds != target.data)

        epoch_acc = training_corrects / (training_corrects + training_incorrects)

        print('training Loss: {:.4f} Acc: {:.4f}'.format(np.mean(train_loss), epoch_acc))

        # evaluation part
        model.eval()
        for data, target in validloader:

            data = data.to(dev)
            target = target.to(dev)

            data = data.view(-1, 306, 250).permute(0, 2, 1)  # (batch, seq, feature)

            output = model(data)

            _, preds = torch.max(output.data, 1)

            loss = loss_function(output, target)
            valid_loss.append(loss.item())

            # statistics
            valid_corrects += torch.sum(preds == target.data)
            valid_incorrects += torch.sum(preds != target.data)

        epoch_acc = valid_corrects / (valid_corrects+valid_incorrects)

        print('validation Loss: {:.4f} Acc: {:.4f}'.format(np.mean(valid_loss), epoch_acc))

        #print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))





