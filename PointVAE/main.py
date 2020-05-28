import os.path as osp
from torch_geometric.datasets import ModelNet, ShapeNet
from torch.nn import Sequential as Seq, Linear as Lin, ReLU,LeakyReLU ,BatchNorm1d as BN, Dropout
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import radius_graph , global_max_pool, knn_graph
from neuralnet_pytorch.metrics import chamfer_loss
from torch_geometric.nn.inits import reset
import torch.nn.functional as F
from layer import MuConv, SigmaConv
from Data import ShapeNet_2048


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, ratio, k):
        super(Net, self).__init__()
        self.k = k
        self.fc1 = MLP([3,  64])
        self.mu_conv = MuConv(MLP([64, 64]),MLP([64,128]))
        self.sig_conv = SigmaConv(MLP([64,64]),MLP([64,128]))
        self.fc3 = Lin(128, 256)
        self.fc4 = Lin(256, 3*2048)
        
    def reset_parameters(self):
        reset(self.fc1)
        reset(self.fc3)
        reset(self.fc4)
        
    def forward(self, data):
        x , pos , batch = data.x , data.x, data.batch
        edge_index = knn_graph(x, self.k, batch, loop=False)
        x = self.fc1(x)
        
        
        z_mu = self.mu_conv(x, pos, edge_index)
        z_sig = self.sig_conv(x, pos,edge_index)
        z_mu = global_max_pool(z_mu, batch)
        z_sig = global_max_pool(z_sig, batch)
        
        z_sig = z_sig.clamp(max=3)
        if self.training:
            z = z_mu + torch.randn_like(z_sig) * torch.exp(z_sig)
        else:
            z = z_mu
        out = F.relu(self.fc3(z))
        out = self.fc4(out)
        out = out.reshape((2048*data.category.size(0),3))
        return out, z_mu, z_sig, z


#path = osp.join(osp.dirname(osp.realpath(__file__)), '..',  'data/ModelNet10')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..',  'data/ShapeNet_2048')
#pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
#train_dataset = ModelNet(path, '10', True, transform, pre_transform)
#test_dataset = ModelNet(path, '10', False, transform, pre_transform)
#train_loader = DataLoader(
#    train_dataset, batch_size=10, shuffle=True, num_workers=1)
#test_loader = DataLoader(
#    test_dataset, batch_size=10, shuffle=False, num_workers=1)
dataset = ShapeNet_2048(path, split='trainval',categories='Chair')
train_loader = DataLoader(
    dataset, batch_size=10, shuffle=True, num_workers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(0.5, 20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = 0
    step = 0
    for data in train_loader:
        step += 1
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        EMD = chamfer_loss(out[0], data.x)
        KLD = -0.5 * torch.mean(
            torch.sum(1 + out[2] - out[1].pow(2) - out[2].exp(),dim=1))
        loss = EMD + KLD
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


for epoch in range(1, 201):
    loss = train()
    print('Epoch {:03d}, Loss: {:.4f}'.format(
        epoch, loss))

