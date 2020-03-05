# -*- coding: utf-8 -*-
import os
import torch
import random
import pickle
import numpy as np
import torch.nn as nn



data = []
for line in open('./business_user.txt','r'):
    (business, user, rating)=line.split(' ')
    data.append((business, user, rating))

random.shuffle(data)

length = data.__len__()

data_train = data[:int(length * 0.8)]
data_test = data[int(length * 0.8):]

i_train, u_train, r_train = [], [], []
for i in range(len(data_train)):
    i_train.append(int(data_train[i][0]))
    u_train.append(int(data_train[i][1]))
    r_train.append(int(data_train[i][2]))

i_test, u_test, r_test = [], [], []
for i in range(len(data_test)):
    i_test.append(int(data_test[i][0]))
    u_test.append(int(data_test[i][1]))
    r_test.append(int(data_test[i][2]))

u_adj = {}
i_adj = {}
for i in range(len(u_train)):
    if u_train[i] not in u_adj.keys():
        u_adj[u_train[i]] = []
    if i_train[i] not in i_adj.keys():
        i_adj[i_train[i]] = []
    u_adj[u_train[i]].extend([(i_train[i],r_train[i])])
    i_adj[i_train[i]].extend([(u_train[i],r_train[i])])

n_users = 1286
n_items = 2614

ufeature = {}
for i in range(n_users):
    ufeature[i] = [0 for _ in range(n_items)]

ifeature = {}
for i in range(n_items):
    ifeature[i] = [0 for _ in range(n_users)]

for key in u_adj.keys():
    n = u_adj[key].__len__()
    for i in range(n):
        ufeature[key][u_adj[key][i][0]] = u_adj[key][i][1]

for key in i_adj.keys():
    n = i_adj[key].__len__()
    for i in range(n):
        ifeature[key][i_adj[key][i][0]] = i_adj[key][i][1]

ufeature_size = ufeature[0].__len__()
ifeature_size = ifeature[0].__len__()

ufea = []
for key in ufeature.keys():
    ufea.append(ufeature[key])
ufea = torch.Tensor(np.array(ufea, dtype = np.float32))
u2e = nn.Embedding(n_users, ufeature_size)
u2e.weight = torch.nn.Parameter(ufea)

ifea = []
for key in ifeature.keys():
    ifea.append(ifeature[key])
ifea = torch.Tensor(np.array(ifea, dtype = np.float32))
i2e = nn.Embedding(n_items, ifeature_size)
i2e.weight = torch.nn.Parameter(ifea)

with open('./_allData.p', 'wb') as meta:
    pickle.dump((u2e, i2e, u_train, i_train, r_train, u_test, i_test, r_test, u_adj, i_adj), meta)