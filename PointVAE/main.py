import os.path as osp
from torch_geometric.datasets import ModelNet, ShapeNet
from layer import MuConv,SigmaConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import radius_graph , global_max_pool

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, ratio, r):
        super(Net, self).__init__()
        self.r = r
        self.fc1 = MLP([3, 64, 64, 128])
        self.mu_conv = MuConv(MLP([128 , 128, 128, 256]))
        self.sig_conv = SigmaConv(MLP([128 , 128, 128, 256]))

        self.Global_nn_mu = MLP([256 + 3, 256, 512, 1024])
        self.Global_nn_sig = MLP([256 + 3, 256, 512, 1024])




        #self.fc2 =

    def forward(self, data):
        x , pos , batch = data.x , data.pos, data.batch
        # idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius_graph(pos, self.r,
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.fc1(x)
        z_mu = self.mu_conv(x, (pos, pos), edge_index)
        z_sig = self.sig_conv(x,(pos, pos),edge_index)

        z_mu = self.Global_nn_mu(torch.cat([z_mu, pos], dim=1))
        z_sig= self.Global_nn_sig(torch.cat([z_sig, pos], dim=1))

        z_mu = global_max_pool(z_mu, batch)
        z_sig = global_max_pool(z_sig, batch)

        if self.training:
            z = z_mu + torch.randn_like(z_sig) * torch.exp(z_sig)
        else:
            z = z_mu


        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)

        return z, pos, batch







path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ShapeNet')
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
train_dataset = ShapeNet(path, split='trainval', pre_transform=pre_transform, transform=transform)
test_dataset = ShapeNet(path, split='test', pre_transform=pre_transform, transform=transform)

# train_dataset = ModelNet(path, '10', True, transform, pre_transform)
# test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=10, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(0.5,0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = (model(data), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for data in train_loader:
    z = model(data)
