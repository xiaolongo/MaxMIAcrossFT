import torch
from encoder import VariationalGCNEncoder
from models import VGAE
from utils import load_data

# synth = True
# dataset = load_data('topology')
# if synth:
#     data = dataset
# else:
#     data = dataset[0]
dataset, data = load_data('Computers')

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = VGAE(encoder=VariationalGCNEncoder(data.num_features, 512)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model(data)
    loss = model.loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    with torch.no_grad():
        z = model(data)
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask])
    return acc


epoches = 500
best_acc = 0
for epoch in range(1, epoches + 1):
    loss = train()
    acc = test()
    if acc > best_acc:
        best_acc = acc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {best_acc:.4f}')
