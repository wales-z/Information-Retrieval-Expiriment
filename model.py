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
        print(U)
        U_b = self.user_bias(u_id).squeeze()

        I = self.item_emb(i_id)
        I_b = self.item_bias(i_id).squeeze()

        # return (U * I).sum(1)+U_b+I_b+self.mean
        return (U * I).sum(1)+U_b+I_b+self.mean, U, I

    def my_gradient_descent(self, u_id, i_id, e_ui, learning_rate, weight_decay):
        '''
        U & I: tensor, size:(batch_size * embedding_size)
        e_ui: tensor, size:(batch_size * 1)
        e_ui*I & e_ui *U 利用了 tensor 点积的广播机制
        '''
        U = self.user_emb(u_id)
        U_b = self.user_bias(u_id).squeeze()

        I = self.item_emb(i_id)
        I_b = self.item_bias(i_id).squeeze()

        U = U - learning_rate * 2*(-e_ui*I + weight_decay*U)
        I = I - learning_rate * 2*(-e_ui*U + weight_decay*I)
        U_b = U_b - learning_rate * 2*(-e_ui + weight_decay*U_b)
        I_b = I_b - learning_rate * 2*(-e_ui + weight_decay*I_b)


def compute_rating(U, I, U_b, I_b, mean):
    return ((U * I).sum()+U_b+I_b+mean).squeeze(0)
