import torch
from torch.utils.data import Dataset
import torch.nn as nn


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

        # return (U * I).sum(1)+U_b+I_b+self.mean
        return (U * I).sum(1)
    
    def my_gradient_descent(self, loss):
        pass
