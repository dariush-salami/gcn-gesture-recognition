from utils import load_data_ply
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import EdgeRNNCell
from sklearn.model_selection import train_test_split
from GestureRec import GestureRecdata
data,label = load_data_ply()
X_train,X_valid,y_train,y_valid = train_test_split(data,label,test_size=.25)
from torch.utils.data import DataLoader
train_dataset = GestureRecdata(X_train,y_train)
valid_dataset = GestureRecdata(X_valid,y_valid)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False)



np.random.seed(42)
torch.manual_seed(42)

torch.cuda.manual_seed(42)

model = EdgeRNNCell(in_channel=data.shape[-1],n_hids=128,k=4,time_steps=data.shape[1],numClasses=np.max(label)+1)

optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)


def train():
    model.train()

    total_loss = 0
    for batch_data ,batch_label in train_loader:
        batch = torch.arange(batch_data.shape[0]).repeat_interleave(2048).to(device)
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        out = model(batch_data,batch)
        loss = F.nll_loss(out, batch_label.flatten())
        loss.backward()
        total_loss += loss.item() * 20
        optimizer.step()
        print(total_loss)
    return total_loss / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data,label in loader:
        data = data.to(device)
        batch = torch.arange(data.shape[0]).repeat_interleave(2048).to(device)
        with torch.no_grad():
            pred = model(data,batch).max(dim=1)[1]
        correct += pred.eq(label).sum().item()
    return correct / len(loader.dataset)

for epoch in range(1, 201):
    loss = train()
    test_acc = test(test_loader)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))