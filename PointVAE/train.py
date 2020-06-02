import os.path as osp
from torch_geometric.datasets import ModelNet, ShapeNet
from torch.nn import Sequential as Seq, Linear as Lin, ReLU,LeakyReLU ,BatchNorm1d as BN, Dropout
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import radius_graph , global_max_pool, knn_graph, global_mean_pool
from neuralnet_pytorch.metrics import chamfer_loss
from torch_geometric.nn.inits import reset
import torch.nn.functional as F
from layer import MuConv, SigmaConv
from Data import ShapeNet_2048


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, k):
        super(Net, self).__init__()
        self.k = k
        self.fc1 = MLP([3, 64, 128])
        self.mu_conv = MuConv(global_nn=MLP([128, 256, 128]))
        self.sig_conv = SigmaConv(global_nn=MLP([128, 256, 128]))

        self.fc3 = Lin(128, 256)
        self.fc4 = Lin(256, 3 * 2048)

    def reset_parameters(self):
        reset(self.fc1)
        reset(self.fc3)
        reset(self.fc4)

    def encode(self, x, batch):
        h1 = self.fc1(x)
        edge_index = knn_graph(h1, self.k, batch, loop=False)
        return global_mean_pool(self.mu_conv(h1, edge_index), batch), global_mean_pool(self.sig_conv(h1, edge_index),
                                                                                       batch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, data):
        x, batch = data.x, data.batch
        mu, logvar = self.encode(x, batch)
        #         print('MU',mu)
        #         print('log', logvar)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


path = osp.join(osp.dirname(osp.realpath(__file__)), '..',  'data/ShapeNet_2048')
dataset = ShapeNet_2048(path, split='trainval',categories='Chair')
train_loader = DataLoader(
    dataset, batch_size=10, shuffle=True, num_workers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)


def train():
    model.train()
    total_loss = 0
    step = 0
    for data in train_loader:
        step += 1
        data = data.to(device)
#         print(data.x.shape)
        optimizer.zero_grad()
        out = model(data)
        CHM = chamfer_loss(out[0].reshape(data.x.shape), data.x)
        KLD = -0.5 * torch.mean(
            torch.sum(1 + out[2] - out[1].pow(2) - out[2].exp(),dim=1))
        if step %100 == 0:
            print('KLD:',KLD)
            print('CHM:',CHM)
        loss = CHM + KLD
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
#         del data
    print('KLD:',KLD)
    print('CHM:',CHM)
    return total_loss / len(dataset)


for epoch in range(1, 401):
    loss = train()
    scheduler.step()
    #test_acc = test(test_loader)
    print('Epoch {:03d}, Loss: {:.4f}'.format(
        epoch, loss))
    if epoch % 10 ==0:
        torch.save(model.state_dict(),'./pointVAEChShape'+'{}'.format(epoch)+'.pt')