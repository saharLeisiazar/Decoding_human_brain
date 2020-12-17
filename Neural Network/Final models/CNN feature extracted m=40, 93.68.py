
# CNN with 7 layers -> train: 99.85	validation: 94.11

from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from scipy.io import loadmat
import os.path

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(7))
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        out = self.fc(out)
        return out


def loadData(fileNameX, fileNameY, fileNameX_test, fileNameY_test):

    dataX = loadmat(fileNameX, squeeze_me=True)
    X = dataX['Training_x']
    dataY = loadmat(fileNameY, squeeze_me=True)
    Y = dataY['Training_y']

    dataX_test = loadmat(fileNameX_test, squeeze_me=True)
    X_test = dataX_test['Testing_x']
    dataY_test = loadmat(fileNameY_test, squeeze_me=True)
    Y_test = dataY_test['Testing_y']

    # Normalize each measurement
    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0)

    X_test = X_test - np.mean(X_test, axis=0)
    X_test = X_test / np.std(X_test, axis=0)

    return X, Y, X_test, Y_test


if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True

    dataPath = './feature extraction/'

    fileNameX = os.path.join(dataPath, "X.mat")
    fileNameY = os.path.join(dataPath, "Y.mat")
    fileNameX_test = os.path.join(dataPath, "X_test.mat")
    fileNameY_test = os.path.join(dataPath, "Y_test.mat")

    X, Y, X_test, Y_test = loadData(fileNameX=fileNameX, fileNameY=fileNameY,
                                    fileNameX_test=fileNameX_test, fileNameY_test=fileNameY_test)

    # create iterator objects for train and valid datasets
    x_tr = torch.tensor(X, dtype=torch.float)
    y_tr = torch.tensor(Y, dtype=torch.long)
    train = TensorDataset(x_tr, y_tr)
    trainloader = DataLoader(train, batch_size=256, shuffle=True)

    x_val = torch.tensor(X_test, dtype=torch.float)
    y_val = torch.tensor(Y_test, dtype=torch.long)
    valid = TensorDataset(x_val, y_val)
    validloader = DataLoader(valid, batch_size=256, shuffle=True)

    model = Model().to(dev)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-9, momentum=0.9)

    best_accuracy = -1
    for epoch in range(0, 100):  # run the model for 100 epochs
        train_loss, valid_loss = [], []

        training_corrects, valid_corrects = 0.0, 0.0
        training_incorrects, valid_incorrects = 0.0, 0.0
        print("epoch : ", epoch)
        # training part
        model.train()
        for data, target in trainloader:
            data = data.view(-1, 1, 70).to(dev)
            target = target.to(dev)

            # 1. forward propagation
            optimizer.zero_grad()
            output = model(data)

            # 2. loss calculation
            loss = loss_function(output, target)
            _, preds = torch.max(output.data, 1)

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
            data = data.view(-1, 1, 70).to(dev)
            target = target.to(dev)

            output = model(data)

            _, preds = torch.max(output.data, 1)
            loss = loss_function(output, target)
            valid_loss.append(loss.item())

            # statistics
            valid_corrects += torch.sum(preds == target.data)
            valid_incorrects += torch.sum(preds != target.data)

        epoch_acc = valid_corrects / (valid_corrects + valid_incorrects)
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc

        print('validation Loss: {:.4f} Acc: {:.4f}'.format(np.mean(valid_loss), epoch_acc))

    print("best accuracy: ", best_accuracy)
