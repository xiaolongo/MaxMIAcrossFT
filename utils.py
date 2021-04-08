import math
import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import (PPI, Amazon, CitationFull, Planetoid,
                                      Reddit)
from torch_geometric.utils import add_remaining_self_loops


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def load_planetoid(dataset):
    data_name = ['Cora', 'CiteSeer', 'PubMed']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Datasets',
                    'NodeData')
    transforms = T.Compose([T.AddSelfLoops()])
    dataset = Planetoid(path, dataset, transform=transforms)
    return dataset, dataset[0]


def load_ppi(dataset):
    data_name = ['PPI']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Datasets',
                    'NodeData', 'PPI')
    dataset = PPI(path)
    return dataset


def load_reddit(dataset):
    data_name = ['Reddit']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Datasets',
                    'NodeData', 'Reddit')
    dataset = Reddit(path)
    return dataset


def load_amazon(dataset):
    data_name = ['Computers', 'Photo']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Datasets',
                    'NodeData')
    # transforms = T.Compose([T.NormalizeFeatures()])
    dataset = Amazon(path, dataset)

    num_per_class = 20
    train_index = []
    test_index = []
    for i in range(dataset.num_classes):
        index = (dataset[0].y.long() == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        if len(index) > num_per_class + 30:
            train_index.append(index[:num_per_class])
            test_index.append(index[num_per_class:])
        else:
            continue
    train_index = torch.cat(train_index)
    test_index = torch.cat(test_index)

    train_mask = index_to_mask(train_index, size=dataset[0].num_nodes)
    test_mask = index_to_mask(test_index, size=dataset[0].num_nodes)

    data = Data(x=dataset[0].x,
                edge_index=dataset[0].edge_index,
                train_mask=train_mask,
                test_mask=test_mask,
                y=dataset[0].y)
    return dataset, data


def load_citation(dataset):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Datasets',
                    'NodeData', 'Citation')
    # transforms = T.Compose([T.NormalizeFeatures()])
    if dataset == 'PubMedFull':
        dataset = 'PubMed'
    dataset = CitationFull(path, dataset)

    num_per_class = 20
    train_index = []
    test_index = []
    for i in range(dataset.num_classes):
        index = (dataset[0].y.long() == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        if len(index) > num_per_class + 30:
            train_index.append(index[:num_per_class])
            test_index.append(index[num_per_class:])
        else:
            continue
    train_index = torch.cat(train_index)
    test_index = torch.cat(test_index)

    train_mask = index_to_mask(train_index, size=dataset[0].num_nodes)
    test_mask = index_to_mask(test_index, size=dataset[0].num_nodes)

    data = Data(x=dataset[0].x,
                edge_index=dataset[0].edge_index,
                train_mask=train_mask,
                test_mask=test_mask,
                y=dataset[0].y)
    return dataset, data


def load_synthdata(dataset):
    data_name = ['feature', 'topology']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'synthdata', dataset)
    x = torch.load(osp.join(path, 'x.pt'))
    edge_index = torch.load(osp.join(path, 'edge_index.pt'))
    edge_index, _ = add_remaining_self_loops(edge_index)
    y = torch.load(osp.join(path, 'y.pt'))

    num_per_class = 20
    train_index, test_index = [], []
    for i in range(3):
        index = (y.long() == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        train_index.append(index[:num_per_class])
        test_index.append(index[num_per_class:])
    train_index = torch.cat(train_index)
    test_index = torch.cat(test_index)

    train_mask = index_to_mask(train_index, size=x.size()[0])
    test_mask = index_to_mask(test_index, size=x.size()[0])

    data = Data(x=x,
                edge_index=edge_index,
                train_mask=train_mask,
                test_mask=test_mask,
                y=y)
    return data


def load_data(dataset):

    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = load_planetoid(dataset)
    elif dataset in ['PPI']:
        data = load_ppi(dataset)
    elif dataset in ['Reddit']:
        data = load_reddit(dataset)
    elif dataset in ['Computers', 'Photo']:
        data = load_amazon(dataset)
    elif dataset in ['PubMedFull']:
        data = load_citation(dataset)
    else:
        data = load_synthdata(dataset)
    return data


if __name__ == '__main__':
    dataset, data = load_data('PubMedFull')
    print(data.num_edge)
    # edge_index = dataset.edge_index.t()
    # edge_index = edge_index.numpy()
    # np.savetxt('./edgelist.txt', edge_index, fmt='%i', delimiter=' ')
