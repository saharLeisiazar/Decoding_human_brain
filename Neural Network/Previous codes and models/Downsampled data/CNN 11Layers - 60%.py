
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

        # 59049 x 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(306, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 19683 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))
        # 6561 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 2187 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU())
        # 729 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),)
            #nn.MaxPool1d(3, stride=3))
        # 243 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),)
            #nn.MaxPool1d(3, stride=3))
        # 81 x 256
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),)
            #nn.MaxPool1d(3, stride=3))
        # 27 x 256
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),)
            #nn.MaxPool1d(3, stride=3))
        # 9 x 256
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),)
            #nn.MaxPool1d(3, stride=3))
        # 3 x 256
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),)
            #nn.MaxPool1d(3, stride=3))
        # 1 x 512
        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(22, stride=22))
        # 1 x 512
        self.fc = nn.Linear(512, 50)
        self.activation = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        return logit


def loadData(filename, downsample=8,start=0,stop=375, numSensors=306):

    data = loadmat(filename, squeeze_me=True)
    X = data['X']
    y = data['y']

    X = signal.resample(X, int(X.shape[2]/downsample), axis=2)
    startIdx = int(start / float(downsample) + 0.5)
    stopIdx = int(stop / float(downsample) + 0.5)
    X = X[..., startIdx:stopIdx]

    # Normalize each measurement

    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0)


    return X, y

if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True

    downsample = 8
    start = 0
    stop = 375
    numSensors = 306
    datapath = '..\decoding-the-human-brain\mats'
    subjects_train = range(1, 17)  # use range(1, 17) for all subjects
    print("Training on subjects", subjects_train)


    X_train = []  # The training data
    y_train = []  # Training labels
    X_test = []  # Test data

    print("Loading %d train subjects." % (len(subjects_train)))

    for subject in subjects_train:
        filename = os.path.join(datapath, "train_subject%02d.mat" % subject)

        XX, yy = loadData(filename=filename,
                               downsample=downsample,
                               start=start,
                               stop=stop)
        print("xx train: ", XX.shape)
        X_train.append(XX)
        y_train.append(yy)

    X = np.vstack(X_train)
    Y = np.concatenate(y_train)
    print("x shape: ", X.shape)
    # create training and validation split
    split_size = int(0.8 * len(X))
    index_list = list(range(len(X)))
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
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-2, momentum=0.9)

    for epoch in range(0, 100):  # run the model for 10 epochs
        train_loss, valid_loss = [], []
        training_corrects, valid_corrects = 0.0, 0.0
        training_incorrects, valid_incorrects = 0.0, 0.0
        print("epoch : ", epoch)
        # training part, 0.0
        model.train()
        for data, target in trainloader:
            target = target.to(dev)

            optimizer.zero_grad()
            data = data.view(-1, 306, 46).to(dev)

            # 1. forward propagation
            #data = data.permute(0, 2, 1) # (batch, seq, feature)
            # print("data: ", data[0, 0, 0:10])
            output = model(data)

            # print(" training output:", output)
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

            data = data.view(-1, 306, 46).to(dev)

            #data = data.permute(0, 2, 1)  # (batch, seq, feature)
            # data = data.view(-1, 306, 375).permute(0, 2, 1)  # (batch, seq, feature)
            output = model(data)
            # print(" validation output:", output)
            _, preds = torch.max(output.data, 1)

            loss = loss_function(output, target)
            valid_loss.append(loss.item())

            # statistics
            valid_corrects += torch.sum(preds == target.data)
            valid_incorrects += torch.sum(preds != target.data)

        epoch_acc = valid_corrects / (valid_corrects + valid_incorrects)

        print('validation Loss: {:.4f} Acc: {:.4f}'.format(np.mean(valid_loss), epoch_acc))

        # print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
