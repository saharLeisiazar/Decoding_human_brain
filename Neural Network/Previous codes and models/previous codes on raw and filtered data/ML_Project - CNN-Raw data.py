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
        self.cnns = nn.Sequential(
            nn.BatchNorm1d(num_features=306),
            nn.Conv1d(in_channels=306, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=250)
            #nn.AvgPool1d(kernel_size=250) # time
        )
        #self.bn = nn.BatchNorm1d(num_features=306)

    def forward(self, input):
        #input = input*10000000
        # input = self.bn(input)
        output = self.cnns(input)
        output = output.view(-1, 1024)  # time :250
        FC = torch.nn.Linear(1024, 2).to(dev)  # fully connected laye
        output = FC(output)
        return output



if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.backends.cudnn.deterministic = True

    path = r'.\data\trainingData'  # use your path
    all_files = glob.glob(path + "/*.csv")
    print("all train files")
    print(all_files)
    #path_to_test_data = r'.\data\testData'  # use your path
    #all_test_files = glob.glob(path_to_test_data + "/*.csv")
    print("all test files")
    #print(all_test_files)

    df = pd.concat((pd.read_csv(f) for f in all_files))  # to read only 5 rows you can write f, nrows=5
    print(df.iloc[0])
    #df_test = pd.concat((pd.read_csv(f_test) for f_test in all_test_files))
    #print(df_test[["y "]])  # this worked
    Y = df.pop("y ")
    X = df
    print("x shpe:", X.shape)
    print("x [0.0]:", X.values[0, 0])
    #Y_test = df_test.pop("y ")
    #X_test = df_test

    # creating tensor from DataFrame dataset
    X = torch.tensor(X.values)
    Y = torch.tensor(Y.values)
    #X_test = torch.tensor(X_test.values)
    #Y_test = torch.tensor(Y_test.values)

    # create training and validation split
    split_size = int(0.8 * len(df))
    index_list = list(range(len(df)))
    train_idx, valid_idx = index_list[:split_size], index_list[split_size:]

    # create iterator objects for train and valid datasets
    x_tr = torch.tensor(X[:split_size], dtype=torch.float)
    y_tr = torch.tensor(Y[:split_size], dtype=torch.long)
    train = TensorDataset(x_tr, y_tr)
    trainloader = DataLoader(train, batch_size=16, shuffle=False)

    x_val = torch.tensor(X[split_size:], dtype=torch.float)
    y_val = torch.tensor(Y[split_size:], dtype=torch.long)
    valid = TensorDataset(x_val, y_val)
    validloader = DataLoader(valid, batch_size=16, shuffle=False)

    model = Model().to(dev)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True,
                                                     threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0.0,
                                                     eps=1e-08)

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
            data = data.view(-1, 306, 375)  # (batch, seq, feature)

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

            data = data.view(-1, 306, 375)  # (batch, seq, feature)

            output = model(data)

            _, preds = torch.max(output.data, 1)

            loss = loss_function(output, target)
            valid_loss.append(loss.item())

            # statistics
            valid_corrects += torch.sum(preds == target.data)
            valid_incorrects += torch.sum(preds != target.data)

        scheduler.step(np.mean(valid_loss))
        epoch_acc = valid_corrects / (valid_corrects+valid_incorrects)

        print('validation Loss: {:.4f} Acc: {:.4f}'.format(np.mean(valid_loss), epoch_acc))


        #print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))





