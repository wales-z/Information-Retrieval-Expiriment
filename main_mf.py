import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import mf, mfDataset


file_path = './ml-1m/ratings.dat'
batch_size = 2048
device = torch.device('cuda:0')
learning_rate = 1e-4
weight_decay = 3e-5
epochs = 100


def main():
    df = pd.read_csv(file_path, header=None, delimiter='::')
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # total dataset, 0.8 for train, 0.2 for val+test
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2)
    # val+test part, 0.5 for val, 0.5 for test
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5)

    train_dataset = mfDataset(np.array(x_train[0]), np.array(
            x_train[1]),  np.array(y_train).astype(np.float32))
    val_dataset = mfDataset(np.array(x_val[0]), np.array(
            x_val[1]),  np.array(y_val).astype(np.float32))
    test_dataset = mfDataset(np.array(x_test[0]), np.array(
            x_test[1]), np.array(y_test).astype(np.float32))

    train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    mean_rating = df.iloc[:, 2].mean()

    num_users = max(df[0])+1
    num_items = max(df[1])+1

    model = mf(num_users, num_items, mean_rating).to(device)
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss().to(device)

    for epoch in range(epochs):
        model.train()
        total_loss, total_len = 0, 0
        for x_u, x_i, y in train_DataLoader:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pre = model(x_u, x_i)
            loss = loss_func(y_pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*len(y)
            total_len += len(y)
        train_loss = total_loss/total_len

        model.eval()
        labels, predicts = [], []
        with torch.no_grad():
            for x_u, x_i, y in test_DataLoader:
                x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
                y_pre = model(x_u, x_i)
                labels.extend(y.tolist())
                predicts.extend(y_pre.tolist())
        mse = mean_squared_error(np.array(labels), np.array(predicts))

        print("epoch {}, train loss is {}, val mse is {}".format(
            epoch, train_loss, mse))

    for x_u, x_i, y in val_DataLoader:
        x_u, x_i, y = x_u, x_i, y
        y_pre = model(x_u, x_i)
        labels.extend(y.tolist())
        predicts.extend(y_pre.tolist())

    mse = mean_squared_error(np.array(labels), np.array(predicts))
    print("test mse loss:{}".format(mse))


if __name__ == '__main__':
    main()
