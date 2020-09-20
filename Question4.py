import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

from Net import Net

epochs = 5000
dataset = np.load('dataset.npy', allow_pickle=True)
label = np.load('label.npy', allow_pickle=True)
train_ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train_data_list = []
test_data_list = []
train_label_list = []
test_label_list = []
positive_rates = []
for train_range in train_ranges:
    for i in range(5):
        sub_data_set = dataset[i]
        sub_label_set = label[i]
        train_size = int(len(sub_data_set) * train_range)
        for j in range(train_size):
            train_data_list.append(sub_data_set[j])
            train_label_list.append(sub_label_set[j][0])
        for j in range(train_size, len(sub_data_set)):
            test_data_list.append(sub_data_set[j])
            test_label_list.append(sub_label_set[j][0])
    train_data = np.array(train_data_list)
    test_data = np.array(test_data_list)
    train_scaler = StandardScaler()
    train_scaler = train_scaler.fit(train_data)
    train_data = train_scaler.transform(train_data)
    test_scaler = StandardScaler()
    test_scaler = test_scaler.fit(test_data)
    test_data = test_scaler.transform(test_data)
    train_label = np.array(train_label_list) - 2
    test_label = np.array(test_label_list) - 2
    train_set = torch.tensor(train_data, dtype=torch.float32)
    target = torch.tensor(train_label)
    test_set = torch.tensor(test_data, dtype=torch.float32)
    ground_truth = torch.tensor(test_label)
    # train_data = np.load('train_set.npy', allow_pickle=True)
    # train_label = np.load('train_label.npy', allow_pickle=True)
    # test_data = np.load('test_set.npy', allow_pickle=True)
    # test_label = np.load('test_label.npy', allow_pickle=True)
    # model = Net()
    # train_set = torch.tensor(train_data, dtype=torch.float32)
    # test_set = torch.tensor(test_data, dtype=torch.float32)
    # ground_truth = torch.tensor(test_label) - 2
    # ground_truth = ground_truth.view([1, ground_truth.shape[0]])[0]
    # net_loss = nn.CrossEntropyLoss()
    # target = torch.tensor(train_label)
    # target = target.view([1, target.shape[0]])[0] - 2
    model = Net()
    net_loss = nn.CrossEntropyLoss()
    train_set = train_set.reshape(train_set.shape[0], 1, train_set.shape[1])
    for e in range(epochs):
        loss = net_loss(model(train_set), target)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train_Range:{} Epoch: {}, Loss: {:.5f}'.format(train_range, e + 1, loss.data))
    model.eval()
    test_set = test_set.reshape(test_set.shape[0], 1, test_set.shape[1])
    output = F.softmax(model(test_set), dim=1)
    pred = output.data.max(1, keepdim=True)[1]
    pred = pred.view([1, pred.shape[0]])[0]
    result = (pred == ground_truth).type(torch.FloatTensor)
    positive_rate = result.sum() / result.shape[0]
    positive_rates.append(positive_rate)
np.save('positive_rates', positive_rates)
