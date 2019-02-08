import torch
import torch.nn as nn


class InputLayer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(InputLayer, self).__init__()

        self.linear1_1 = nn.Linear(channels_in, channels_out)
        self.linear1_2 = nn.Linear(channels_in, channels_out)

        self.linear2_1 = nn.Linear(channels_out, channels_out)
        self.linear2_2 = nn.Linear(channels_out, channels_out)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.linear1_1(x) + self.linear1_2(x))
        x2 = self.relu(self.linear2_1(x1) + self.linear2_2(x1))

        return x2


class EncoderLayer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(EncoderLayer, self).__init__()

        self.linear1_1 = nn.Linear(channels_in, channels_out)
        self.linear1_2 = nn.Linear(channels_in, channels_out)

        self.linear2_1 = nn.Linear(channels_out, channels_out)
        self.linear2_2 = nn.Linear(channels_out, channels_out)

        self.relu = nn.ReLU()

    def forward(self, x, adj_mat):
        x1 = self.relu(self.linear1_1(x) + torch.mm(adj_mat, self.linear1_2(x)))
        x2 = self.relu(self.linear2_1(x1) + self.linear2_2(x1))

        return x2


class GraphNN(nn.Module):
    def __init__(self, features_in=1, features_out=1, num_channels=8):
        super(GraphNN, self).__init__()

        self.layer1 = EncoderLayer(features_in, num_channels)
        self.layer2 = EncoderLayer(num_channels, num_channels)
        self.layer3 = EncoderLayer(num_channels, num_channels)
        self.layer4 = EncoderLayer(num_channels, features_out)

    def forward(self, adj_mat, feature_mat):
        norm_deg_mat = self.norm_deg_mat(adj_mat)
        x1 = self.layer1(feature_mat, norm_deg_mat)
        x2 = self.layer2(x1, norm_deg_mat)
        x3 = self.layer3(x2, norm_deg_mat)
        x4 = self.layer4(x3, norm_deg_mat)

        return x4

    # assume adj_mat is torch.sparse.FloatTensor
    def norm_deg_mat(self, adj_mat):
        num_nodes = adj_mat.shape[0]
        degrees = torch.sparse.sum(adj_mat, dim=1).to_dense()
        print(degrees)
        diag_idx = torch.arange(num_nodes).repeat(2, 1)
        print(diag_idx)
        diag = torch.sparse.FloatTensor(diag_idx, 1 / degrees, torch.Size([num_nodes, num_nodes]))
        norm_deg_mat = torch.sparse.mm(diag, adj_mat.to_dense())

        return norm_deg_mat
        

                
