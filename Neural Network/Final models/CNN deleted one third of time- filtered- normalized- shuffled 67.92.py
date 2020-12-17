#67.82
# CNN with 11 layers -> train: 91.85	validation: 60.17

from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from scipy.io import loadmat
from scipy import signal
import os.path


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(306, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        out = self.fc(out)
        return out


def loadData(fileName):

    data = loadmat(fileName, squeeze_me=True)
    X = data['X']
    y = data['y']
    # Sort sensors ordered by their location (from back to front)
    # The mat file is generated from file NeuroMagSensorsDeviceSpace.mat
    # provided by the organizers.
    sensorLocations = loadmat("SensorsSortedLocation.mat")

    index = sensorLocations["SensorsSorted"] - 1
    index = index.ravel()
    index = index[:306]
    X = X[:, index, :]

    # Extract only the 2/3 of temporal dimension
    X = X[..., 125:]

    # Normalize each trial
    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0)

    # Apply FIR filter
    filter = signal.firwin(400, [0.01, 0.08], pass_zero=False)
    for i in range(len(X)):
        for j in range(306):
            X[i, j, :] = signal.convolve(X[i, j, :], filter, mode='same')

    return X, y


if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True

    dataPath = '..\decoding-the-human-brain\mats'
    subjects_train = range(1, 17)  # use range(1, 17) for all subjects
    X, Y = [], []

    for subject in subjects_train:
        fileName = os.path.join(dataPath, "train_subject%02d.mat" % subject)
        X_loaded, Y_loaded = loadData(fileName=fileName)
        X.append(X_loaded)
        Y.append(Y_loaded)

    X = np.vstack(X)
    Y = np.concatenate(Y)

    split_size = int(0.8 * len(X))
    index_list = list(range(len(X)))
    train_idx, valid_idx = index_list[:split_size], index_list[split_size:]

    # create iterator objects for train and valid datasets
    x_tr = torch.tensor(X[:split_size], dtype=torch.float)
    y_tr = torch.tensor(Y[:split_size], dtype=torch.long)
    train = TensorDataset(x_tr, y_tr)
    trainloader = DataLoader(train, batch_size=128, shuffle=True)

    x_val = torch.tensor(X[split_size:], dtype=torch.float)
    y_val = torch.tensor(Y[split_size:], dtype=torch.long)
    valid = TensorDataset(x_val, y_val)
    validloader = DataLoader(valid, batch_size=128, shuffle=True)

    model = Model().to(dev)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-2, momentum=0.9)

    for epoch in range(0, 100):  # run the model for 100 epochs
        train_loss, valid_loss = [], []
        training_corrects, valid_corrects = 0.0, 0.0
        training_incorrects, valid_incorrects = 0.0, 0.0
        print("epoch : ", epoch)
        # training part
        model.train()
        for data, target in trainloader:
            data = data.view(-1, 306, 250).to(dev)
            target = target.to(dev)

            # 1. forward propagation
            optimizer.zero_grad()
            output = model(data)

            # 2. loss calculation
            _, preds = torch.max(output.data, 1)
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
            data = data.view(-1, 306, 250).to(dev)
            target = target.to(dev)

            output = model(data)

            _, preds = torch.max(output.data, 1)
            loss = loss_function(output, target)
            valid_loss.append(loss.item())

            # statistics
            valid_corrects += torch.sum(preds == target.data)
            valid_incorrects += torch.sum(preds != target.data)

        epoch_acc = valid_corrects / (valid_corrects + valid_incorrects)
        print('validation Loss: {:.4f} Acc: {:.4f}'.format(np.mean(valid_loss), epoch_acc))