import os.path as osp
from torch_geometric.datasets import ModelNet, ShapeNet
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d as BN, Dropout
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import radius_graph, global_max_pool
from neuralnet_pytorch.metrics import chamfer_loss
from torch_geometric.nn.inits import reset
import torch.nn.functional as F
from layer import MuConv, SigmaConv
import pickle

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, ratio, r):
        super(Net, self).__init__()
        self.r = r
        self.fc1 = MLP([3, 64])
        self.mu_conv = MuConv(MLP([64, 64]), MLP([64, 20]))
        self.sig_conv = SigmaConv(MLP([64, 64]), MLP([64, 20]))
        self.fc3 = Lin(20, 1024)
        self.fc4 = Lin(1024, 3072)

    def reset_parameters(self):
        reset(self.fc1)
        reset(self.fc3)
        reset(self.fc4)

    def forward(self, data):
        x, pos, batch = data.pos, data.pos, data.batch
        edge_index = radius_graph(pos, self.r,
                                  max_num_neighbors=64)
        x = self.fc1(x)
        z_mu = self.mu_conv(x, pos, edge_index)
        z_sig = self.sig_conv(x, pos, edge_index)
        z_mu = global_max_pool(z_mu, batch)
        z_sig = global_max_pool(z_sig, batch)
        z_sig = z_sig.clamp(max=10)
        if self.training:
            z = z_mu + torch.randn_like(z_sig) * torch.exp(z_sig)
        else:
            z = z_mu
        out = F.relu(self.fc3(z))
        out = self.fc4(out)
        out = out.reshape((1024 * data.y.size(0), 3))
        return out, z_mu, z_sig, z


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=10, shuffle=True, num_workers=1)
test_loader = DataLoader(
    test_dataset, batch_size=10, shuffle=False, num_workers=1)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = Net(0.5, 0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load('./trained_model/pointVAECh.pt'))
model.eval()
generated_data = []
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        pred = model(data)
        generated_data.append({
            'pred': pred[0],
            'true': data.pos
        })
        print(pred[0])
with open('augmented_data.pkl', 'wb') as handle:
    pickle.dump(generated_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Model loaded successfully :)')
