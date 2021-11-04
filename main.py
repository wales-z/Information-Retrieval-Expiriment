import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import mfDataset, compute_rating


file_path = './ml-1m/ratings.dat'
batch_size = 1
device = torch.device('cuda:0')
learning_rate = 1e-3
weight_decay = 1e-5
epochs = 100
embedding_size = 100
# loss_func = torch.nn.MSELoss().to(device)


df = pd.read_csv(file_path, header=None, delimiter='::')
x, y = df.iloc[:, :2], df.iloc[:, 2]

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
print(f"num_users:{num_users}")
print(f"num_items:{num_items}")


# create user & item latent factor vectors
user_emb = torch.empty(num_users, embedding_size, dtype=torch.float32)
user_bias = torch.empty(num_users, 1, dtype=torch.float32)
item_emb = torch.empty(num_items, embedding_size, dtype=torch.float32)
item_bias = torch.empty(num_items, 1, dtype=torch.float32)

nn.init.normal_(user_emb, mean=0, std=0.1)
nn.init.normal_(user_bias, mean=0, std=0.1)
nn.init.normal_(item_emb, mean=0, std=0.1)
nn.init.normal_(item_bias, mean=0, std=0.1)


# training & val
for epoch in range(epochs):
    # training phase
    total_loss, total_len = 0, 0
    for x_u, x_i, y in train_DataLoader:
        x_u, x_i, y = x_u, x_i, y
        y_pre = compute_rating(user_emb[x_u], item_emb[x_i], user_bias[x_u], item_bias[x_i], mean_rating)
        loss = torch.pow((y-y_pre), 2) + weight_decay * (torch.sum(torch.pow(user_emb[x_u], 2)) + torch.sum(torch.pow(item_emb[x_i], 2)) + torch.sum(torch.pow(user_bias[x_u], 2)) + torch.sum(torch.pow(item_bias[x_i], 2)))
        e_ui = y-y_pre

        # gradient descent
        user_emb[x_u] -= learning_rate * 2*(-e_ui*item_emb[x_i] + weight_decay*user_emb[x_u])
        item_emb[x_i] -= learning_rate * 2*(-e_ui*user_emb[x_u] + weight_decay*item_emb[x_i])
        user_bias[x_u] -= learning_rate * 2*(-e_ui + weight_decay*user_bias[x_u])
        item_bias[x_i] -= learning_rate * 2*(-e_ui + weight_decay*item_bias[x_i])

        total_loss += loss.item()*len(y)
        total_len += len(y)
    train_loss = total_loss/total_len

    # val phase
    labels, predicts = [], []
    for x_u, x_i, y in val_DataLoader:
        x_u, x_i, y = x_u, x_i, y
        y_pre = compute_rating(user_emb[x_u], item_emb[x_i], user_bias[x_u], item_bias[x_i], mean_rating)
        labels.extend(y.tolist())
        predicts.extend(y_pre.tolist())
    mse = mean_squared_error(np.array(labels), np.array(predicts))

    print("epoch {}, train loss is {}, val mse is {}".format(
        epoch, train_loss, mse))

# test phase
for x_u, x_i, y in val_DataLoader:
    x_u, x_i, y = x_u, x_i, y
    y_pre = compute_rating(user_emb[x_u], item_emb[x_i], user_bias[x_u], item_bias[x_i], mean_rating)
    labels.extend(y.tolist())
    predicts.extend(y_pre.tolist())

mse = mean_squared_error(np.array(labels), np.array(predicts))
print("test mse loss:{}".format(mse))
