import random

import torch
import torch.nn as nn
import torch.optim as optim

from net import GraphNN


def train():

    model = GraphNN()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    training_data = get_training_data()
    loss = 0

    print("Beginning Training...")
    for i in range(100):
        adj_mats, features, labels = get_sample(training_data, 10)
        for ex in range(10):
            output = model(adj_mats[ex], features[ex])
            print(output.shape)
            loss += criterion(output, labels[ex])
        print("Loss: {}".format(loss.item()))
        loss.backward()
        loss = 0
        optimizer.step()
        optimizer.zero_grad()


def get_training_data():
    adj_mats = []
    features_mat = []
    labels_mat = []
    for i in range(100):
        edges = torch.randint(5, size=(2, 10))
        vals = torch.ones(10)
        adj_mat = torch.sparse.FloatTensor(edges, vals, torch.Size([5, 5]))
        features = torch.randn(5, 1)
        labels = torch.randint(5, size=(5,))
        adj_mats.append(adj_mat)
        features_mat.append(features)
        labels_mat.append(labels)
    training_data = list(zip(adj_mats, features_mat, labels_mat))
    return training_data


def get_sample(data, size):
    sample = random.sample(data, size)
    adj_mats, features, labels = zip(*sample)
    return adj_mats, features, labels


if __name__ == "__main__":
    train()
