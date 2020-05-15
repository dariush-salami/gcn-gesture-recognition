import os.path as osp
from torch_geometric.datasets import ModelNet, ShapeNet
from layer import MuConv,SigmaConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import radius_graph , global_max_pool
import matplotlib.pyplot as plt
import numpy as np
from neuralnet_pytorch.metrics import chamfer_loss
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


        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Lin(128, 3))

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


        # pos = pos.new_zeros((x.size(0), 3))
        # batch = torch.arange(x.size(0), device=batch.device)
        #out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        #out = self.mlp(out)

        #decoded = self.lin1(z)
        decoded = self.mlp(z)
        return decoded, z, pos, batch







path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ShapeNet')
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
train_dataset = ShapeNet(path, split='trainval', pre_transform=pre_transform)
test_dataset = ShapeNet(path, split='test', pre_transform=pre_transform)

# train_dataset = ModelNet(path, '10', True, transform, pre_transform)
# test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=10, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(0.5,0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = chamfer_loss(model(data), data.pos)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 201):
    loss = train()
    #test_acc = test(test_loader)
    print('Epoch {:03d}, Loss: {:.4f}'.format(
        epoch, loss))




# for data in train_loader:
#     z = model(data)

# fig = plt.figure()
#
#
# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(train_dataset[3].pos[:,0],train_dataset[3].pos[:,1],train_dataset[3].pos[:,2])

# n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, marker=m)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()