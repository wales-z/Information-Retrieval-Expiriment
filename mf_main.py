import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


file_path = '../data/1m.dat'
batch_size = 2048
device = torch.device('cuda:0')
learning_rate = 1e-4
weight_decay = 3e-5
epochs = 100


class mfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class mf(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):
        super(mf, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)

        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.normal_(0, 0.1)
        self.user_bias.weight.data.normal_(0, 0.1)

        self.item_emb.weight.data.normal_(0, 0.1)
        self.item_bias.weight.data.normal_(0, 0.1)

        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        U_b = self.user_bias(u_id).squeeze()

        I = self.item_emb(i_id)
        I_b = self.item_bias(i_id).squeeze()

        return (U * I).sum(1)+U_b+I_b+self.mean


def main():
    df = pd.read_csv(file_path, header=None, delimiter='::')
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    train_dataset = mfDataset(np.array(x_train[0]), np.array(
        x_train[1]),  np.array(y_train).astype(np.float32))
    test_dataset = mfDataset(np.array(x_test[0]), np.array(
        x_test[1]), np.array(y_test).astype(np.float32))

    train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    mean_rating = df.iloc[:, 2].mean()

    num_users = max(df[0])+1
    num_items = max(df[1])+1

    model = mf(num_users, num_items, mean_rating).to(device)
    optimizer = torch.optim.Adam(
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


if __name__ == '__main__':
    main()
