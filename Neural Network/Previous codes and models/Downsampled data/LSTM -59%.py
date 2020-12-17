
# best accuracy 59.00% epoch 6  -> lr=0.01, weight_decay=1e-6, momentum=0.9 , batch: 64, start = 130
# best accuracy 59.69% epoch 78 -> lr=0.001, weight_decay=1e-3, momentum=0.9 , batch: 64 , start = 130
# best accuracy 59.75% epoch 78 -> lr=0.001, weight_decay=1e-3, momentum=0.9 , batch: 64 , start = 0

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
        self.lstm = nn.LSTM(input_size=306, hidden_size=250, num_layers=2,
                            batch_first=True, bias=False)
        #self.h0 = torch.randn(1, 128, 40)  # 40-> the number of hidden layer features
        self.FC = torch.nn.Linear(250, 2)  # fully connected layer
        #self.bn = nn.BatchNorm1d(num_features=306)

    def forward(self, input):

        # input = self.bn(input)
        h0 = torch.randn(2, input.shape[0], 250)  # 40-> the number of hidden layer features
        c0 = torch.randn(2, input.shape[0], 250)  # 40-> the number of hidden layer features
        h0 = h0.to(dev)
        c0 = c0.to(dev)
        #_, (hn, cn) = self.lstm(input, (h0, c0))
        _, (hn, cn) = self.lstm(input)
        hn = hn[-1:]
        hn = hn.view(-1, 250)

        hn = self.FC(hn)
        return hn



def loadData(filename,downsample=8,start=130, stop=375,numSensors=306 ):

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
    start = 130
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
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)

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
#            data = data.view(-1, 306, 46)

            # 1. forward propagation
            data = data.permute(0, 2, 1).to(dev)  # (batch, seq, feature)
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

            #data = data.view(-1, 306, 46)

            data = data.permute(0, 2, 1).to(dev)  # (batch, seq, feature)
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
