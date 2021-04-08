import argparse

import torch

from encoder import APPNPEncoder, GCNEncoder, SAGEEncoder, corruption, summary
from models import MultiViewDGI
from utils import load_data


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    model(data)
    loss = model.loss()
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model(data)
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask])
    return acc


def train_eval(epoches, model, optimizer, data):
    best_acc = 0
    for epoch in range(1, epoches + 1):
        loss = train(model, data, optimizer)
        acc = test(model, data)
        if acc > best_acc:
            best_acc = acc
            # torch.save(model, './parameter/mvdgi_Computers.pkl')
        if epoch == 300:
            torch.save(model, './parameter/mvdgi_CiteSeer.pkl')
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--epoches', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=512)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}')

    dataset = load_data(args.dataset)
    dataset, data = load_data(args.dataset)
    data = data.to(device)

    assert args.encoder in ['GCN', 'SAGE']
    if args.encoder == 'GCN':
        encoder = GCNEncoder
    else:
        encoder = SAGEEncoder

    model = MultiViewDGI(hidden_dim=args.hidden,
                         encoder=encoder(dataset.num_features, args.hidden),
                         encoder_d=APPNPEncoder(dataset.num_features,
                                                args.hidden),
                         summary=summary,
                         corruption=corruption).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_eval(epoches=args.epoches,
               model=model,
               optimizer=optimizer,
               data=data)
