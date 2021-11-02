# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %% [markdown]
# ## 1 Import libraries or packges that we need

# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import mf, mfDataset

# %% [markdown]
# ## 2 Set the training configuration, i.e. hyper-paramters, computing device, dataset path

# %%
file_path = './ml-1m/ratings.dat'
batch_size = 2048
device = torch.device('cuda:0')
learning_rate = 1e-2
weight_decay = 1e-5
epochs = 15

# %% [markdown]
# ## 3 Preprocessing before training

# %%
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
print(f"num_users:{num_users-1}")
print(f"num_items:{num_items-1}")

# generate two mf model objects: model using pytorch auto-gradient，model_my_SGD using my implementation of gradient descent
model = mf(num_users, num_items, mean_rating).to(device)
model_my_SGD = mf(num_users, num_items, mean_rating).to(device)

# l2 normalization
optimizer = torch.optim.SGD(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# optimizer = torch.optim.Adam(
#         params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_func = torch.nn.MSELoss().to(device)

# %% [markdown]
# ## 4.1 Training and evaluation：pytorch
# optimization method: Stochastic Gradient Descent (SGD) or Adam

# %%
print("pytorch auto-gradient round")
print("__________________________________________________________________")

for epoch in range(epochs):
    # training phase

    model.train()
    total_loss, total_len = 0, 0
    for x_u, x_i, y in train_DataLoader:
        x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
        y_pre, p_u, q_i = model(x_u, x_i)
        loss = loss_func(y_pre, y)

        # auto gradient computing and gradient descent based on pytorch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*len(y)
        total_len += len(y)
    train_loss = total_loss/total_len


    # evaluation phase
    model.eval()
    labels, predicts = [], []
    with torch.no_grad():
        for x_u, x_i, y in test_DataLoader:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pre, p_u, q_i = model(x_u, x_i)
            labels.extend(y.tolist())
            predicts.extend(y_pre.tolist())
    mse = mean_squared_error(np.array(labels), np.array(predicts))

    print("epoch {}, train loss is {}, val mse is {}".format(
        epoch, train_loss, mse))
print("__________________________________________________________________")

# %% [markdown]
# ## 4.2 Training and evaluation：my implemention for gradient descent
# optimization method: Stochastic Gradient Descent (SGD)

# %%
# 重新生成torch的dataloader部分
train_dataset = mfDataset(np.array(x_train[0]), np.array(
        x_train[1]),  np.array(y_train).astype(np.float32))
test_dataset = mfDataset(np.array(x_test[0]), np.array(
        x_test[1]), np.array(y_test).astype(np.float32))

train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("my implementation round")
print("__________________________________________________________________")

for epoch in range(epochs):
    # training phase
    model.train()
    total_loss, total_len = 0, 0
    for x_u, x_i, y in train_DataLoader:
        x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
        y_pre, p_u, q_i = model_my_SGD(x_u, x_i)
        # l2 normalization
        loss = loss_func(y_pre, y) + weight_decay * (torch.sum(torch.pow(p_u, 2)) + torch.sum(torch.pow(q_i, 2)))

        # my implementation for gradient descent of mf model
        e_ui = (y - y_pre).unsqueeze(1)
        model_my_SGD.my_gradient_descent(x_u, x_i, e_ui, learning_rate, weight_decay)

        total_loss += loss.item()*len(y)
        total_len += len(y)
    train_loss = total_loss/total_len

    # evaluation phase
    model.eval()
    labels, predicts = [], []
    with torch.no_grad():
        for x_u, x_i, y in test_DataLoader:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pre, p_u, q_i = model(x_u, x_i)
            labels.extend(y.tolist())
            predicts.extend(y_pre.tolist())
    mse = mean_squared_error(np.array(labels), np.array(predicts))

    print("epoch {}, train loss is {}, val mse is {}".format(
        epoch, train_loss, mse))


