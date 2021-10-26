import torch
import numpy as np
import pandas as pd
import torch.nn as nn

def data_preprocess():
    
    movie_data_raw = pd.read_table('./ml-1m/movies.dat',sep='\t',header=None,encoding='latin-1')
    rating_data_raw = pd.read_table('./ml-1m/ratings.dat',sep='\t',header=None,encoding='latin-1')
    users_data_raw = pd.read_table('./ml-1m/users.dat',sep='\t',header=None,encoding='latin-1')

    d = {'Movie-Id':[],'Title':[],'Year':[],'Type':[]}
    for i in  range(movie_data_raw.values.shape[0]):
        tmp= movie_data_raw.values[i].item().split("::")
        d['Movie-Id'].append(tmp[0])
        d['Title'].append(tmp[1][:-6])
        d['Year'].append(int(tmp[1][-5:-1]))
        d['Type'].append(tmp[2].split('|'))
    movie_data = pd.DataFrame(data=d)

    d = {'User-Id':[],'Gender':[],'Age':[],'Occupation':[],'Zip-code':[]}
    for i in  range(users_data_raw.values.shape[0]):
        tmp= users_data_raw.values[i].item().split("::")
        d['User-Id'].append(tmp[0])
        d['Gender'].append(tmp[1])
        d['Age'].append(int(tmp[2]))
        d['Occupation'].append(tmp[3])
        d['Zip-code'].append(tmp[4])
    user_data = pd.DataFrame(data=d)

    d = {'User-Id':[],'Movie-Id':[],'Rating':[],'Timestamp':[]}
    for i in  range(rating_data_raw.values.shape[0]):
        tmp= rating_data_raw.values[i].item().split("::")
        d['User-Id'].append(tmp[0])
        d['Movie-Id'].append(tmp[1])
        d['Rating'].append(tmp[2])
        d['Timestamp'].append(tmp[3])
    rating_data = pd.DataFrame(data=d)

    hash = {}
    for i in range(movie_data['Movie-Id'].shape[0]):
        hash[movie_data['Movie-Id'][i]] = i

    rating_data = np.array(rating_data)
    rating_data_shuffle = np.random.permutation(rating_data)
    n_sample = rating_data_shuffle.shape[0]
    n_train = int(n_sample * 0.8)
    n_val = int(n_sample * 0.1)
    rating_data_train = rating_data_shuffle[:n_train]
    rating_data_val = rating_data_shuffle[n_train:n_train+n_val]
    rating_data_test = rating_data_shuffle[n_train+n_val:]

    r_mat_train = np.empty((user_data.shape[0],movie_data.shape[0]))  # user x movie
    for i in rating_data_train:
        userid = int(i[0])-1
        movieid = i[1]
        rating = int(i[2])
        
        r_mat_train[userid][hash[movieid]] = rating

    r_mat_test = np.empty((user_data.shape[0],movie_data.shape[0]))  # user x movie
    for i in rating_data_test:
        userid = int(i[0])-1
        movieid = i[1]
        rating = int(i[2])
        
        r_mat_test[userid][hash[movieid]] = rating

    r_mat_val = np.empty((user_data.shape[0],movie_data.shape[0]))  # user x movie
    for i in rating_data_val:
        userid = int(i[0])-1
        movieid = i[1]
        rating = int(i[2])
        
        r_mat_val[userid][hash[movieid]] = rating

    return r_mat_train,r_mat_val,r_mat_test,user_data,movie_data

def compute_MSEloss(r_mat_t,r_mat,p_mat,q_mat,reg):
    # p_mat M x K   q_mat N x K   r_mat M x N
    n = r_mat.shape[0] * r_mat.shape[1]
    error = r_mat_t - r_mat

    
    loss = 0.5*(np.sum((r_mat_t - r_mat)**2)+reg*(np.sum(p_mat**2)+np.sum(q_mat**2)))


    p_grad = -np.dot(error,q_mat) + reg * p_mat
    q_grad = -np.dot(error.T,p_mat) + reg * q_mat

    grad = (p_grad/n,q_grad/n)
    return loss/n, grad

def evaluate(p_mat,q_mat,r_mat):
    r = np.dot(p_mat,q_mat.T)
    m = r_mat.shape[0]
    n = r_mat.shape[1]
    pre = 0
    count = 0
    for i in range(m):
        for j in range(n):
            if r_mat[i][j] !=0 :
                count += 1
                if r_mat[i][j] == round(r[i][j]):
                    pre += 1
    return pre/count
