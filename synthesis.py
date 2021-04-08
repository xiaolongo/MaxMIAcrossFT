import numpy as np
import torch
from scipy import sparse
from scipy.stats import multivariate_normal
from torch_geometric.utils import erdos_renyi_graph, from_scipy_sparse_matrix


def feature_graph():
    edge_index = erdos_renyi_graph(2400, 0.01)
    torch.save(edge_index, './synthdata/feature/edge_index.pt')

    dim = 20
    mask_convariance_maxtix = np.diag([1 for _ in range(dim)])

    center1 = 2.5 * np.random.random(size=dim) - 1
    center2 = 2.5 * np.random.random(size=dim) - 1
    center3 = 2.5 * np.random.random(size=dim) - 1

    data1 = multivariate_normal.rvs(mean=center1,
                                    cov=mask_convariance_maxtix,
                                    size=800)
    data2 = multivariate_normal.rvs(mean=center2,
                                    cov=mask_convariance_maxtix,
                                    size=800)
    data3 = multivariate_normal.rvs(mean=center3,
                                    cov=mask_convariance_maxtix,
                                    size=800)
    data = np.vstack((data1, data2, data3))

    label = np.array([0 for _ in range(800)] + [1 for _ in range(800)] +
                     [2 for _ in range(800)])

    permutation = np.random.permutation(label.shape[0])

    data = data[permutation, :]
    label = label[permutation]
    x, y = torch.from_numpy(data), torch.from_numpy(label)
    x, y = x.float(), y.long()
    torch.save(x, './synthdata/feature/x.pt')
    torch.save(y, './synthdata/feature/y.pt')


def topology_graph():
    adj = np.zeros((2400, 2400))
    for i in range(800):
        for j in range(i + 1, 800):
            z = np.random.randint(0, 100, dtype=int)
            if z > 96:  # 0.03
                adj[i, j] = 1
                adj[j, i] = 1

    for i in range(800, 1600):
        for j in range(i + 1, 1600):
            z = np.random.randint(0, 100, dtype=int)
            if z > 96:  # 0.03
                adj[i, j] = 1
                adj[j, i] = 1

    for i in range(1600, 2400):
        for j in range(i + 1, 2400):
            z = np.random.randint(0, 100, dtype=int)
            if z > 96:  # 0.03
                adj[i, j] = 1
                adj[j, i] = 1

    for i in range(800):
        for j in range(800, 1600):
            z = np.random.randint(0, 10000, dtype=int)
            if z > 9999:  # 0.0001
                adj[i, j] = 1
                adj[j, i] = 1

    for i in range(800):
        for j in range(1600, 2400):
            z = np.random.randint(0, 10000, dtype=int)
            if z > 9999:  # 0.00001
                adj[i, j] = 1
                adj[j, i] = 1

    for i in range(800, 1600):
        for j in range(1600, 2400):
            z = np.random.randint(0, 10000, dtype=int)
            if z > 9999:  # 0.00001
                adj[i, j] = 1
                adj[j, i] = 1
    arr_sparse = sparse.coo_matrix(adj)
    edge_index, _ = from_scipy_sparse_matrix(arr_sparse)
    edge_index = edge_index.long()
    torch.save(edge_index, './synthdata/topology/edge_index.pt')

    dim = 20
    mask_convariance_maxtix = np.diag([1 for _ in range(dim)])

    center1 = 2.5 * np.random.random(size=dim) - 1
    center2 = 2.5 * np.random.random(size=dim) - 1
    center3 = 2.5 * np.random.random(size=dim) - 1

    data1 = multivariate_normal.rvs(mean=center1,
                                    cov=mask_convariance_maxtix,
                                    size=800)
    data2 = multivariate_normal.rvs(mean=center2,
                                    cov=mask_convariance_maxtix,
                                    size=800)
    data3 = multivariate_normal.rvs(mean=center3,
                                    cov=mask_convariance_maxtix,
                                    size=800)
    data = np.vstack((data1, data2, data3))

    label = np.array([0 for _ in range(800)] + [1 for _ in range(800)] +
                     [2 for _ in range(800)])
    x, y = torch.from_numpy(data), torch.from_numpy(label)
    x, y = x.float(), y.long()
    torch.save(x, './synthdata/topology/x.pt')
    torch.save(y, './synthdata/topology/y.pt')


if __name__ == '__main__':
    # feature_graph()
    topology_graph()
