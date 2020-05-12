import os.path as osp
from torch_geometric.datasets import ModelNet
from layer import MuConv,SigmaConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import radius

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nnMU,nnSIG):
        super(SAModule, self).__init__()
        self.r = r
        self.fc1 = MLP([3, 64, 64, 128])
        self.mu_conv = MuConv(nnMU)
        self.sig_conv = SigmaConv(nnSIG)

    def forward(self, x, pos, batch):
        # idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos, self.r, batch, batch,
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        Mu = self.conv(x, (pos, pos), edge_index)

        return x, pos, batch
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=10, shuffle=True, num_workers=0)
test_loader = DataLoader(
    test_dataset, batch_size=10, shuffle=False, num_workers=0)