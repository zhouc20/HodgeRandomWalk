import numpy as np
import torch
from torch_sparse import SparseTensor
from torch.linalg import det
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# compute div(w), outflow is positive
def divergence(adj):
    outflow = torch.sum(adj, dim=1)
    inflow = torch.sum(adj, dim=0)
    div = outflow - inflow
    return div

def Delta_0(alpha, A):
    w_g = A * alpha.unsqueeze(0).repeat(A.shape[-1], 1) - A * alpha.unsqueeze(1).repeat(1, A.shape[-1])
    return w_g


# directly compute graph Laplacian L0 (0-order Hodge Laplacian)
def compute_graph_Laplacian_0(edge_index, Nm=None):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    A1 = (A + A.permute(1, 0)).bool().float()
    D = torch.diag_embed(torch.sum(A1, dim=0))
    L0 = D - A1
    return L0


# edge is directed
def compute_gradient(edge_index, edge_attr, Nm=None):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=edge_attr,
                       sparse_sizes=(Nm, Nm)).coalesce().to_dense()

    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    div = divergence(adj)
    L0 = compute_graph_Laplacian_0(edge_index, Nm)
    alpha = torch.linalg.solve(L0, -div)
    w_g = Delta_0(alpha, A)
    return w_g


def extract_triangle(edge_index, Nm=None, directed=True):
    triangle_set = []
    if directed:
        A = SparseTensor(row=edge_index[0],
                         col=edge_index[1],
                         value=torch.ones(edge_index.shape[1], dtype=torch.float),
                         sparse_sizes=(Nm, Nm)).coalesce().to_dense()
        A = (A + A.permute(1, 0)).bool().float()
        edge_index, _ = dense_to_sparse(A)
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    for i in range(Nm):
        subset, sub_edge_idx, mapping, edge_mask = k_hop_subgraph(i, 1, edge_index, relabel_nodes=False, num_nodes=Nm)
        for j in range(sub_edge_idx.shape[1]):
            if sub_edge_idx[0, j] > i and sub_edge_idx[1, j] > i and sub_edge_idx[0, j] < sub_edge_idx[1, j]:
                tri_node, _ = torch.sort(torch.tensor([i, sub_edge_idx[0, j], sub_edge_idx[1, j]]))
                triangle_set.append(tri_node)
    return triangle_set


def delta_1_delta_1_star(triangle_set):
    d1d1s = torch.diag_embed(torch.ones(len(triangle_set)) * 3.)
    for i in range(len(triangle_set)):
        c = triangle_set[i]
        pc = [torch.tensor([c[0], c[1]]), torch.tensor([c[1], c[2]]), torch.tensor([c[2], c[0]])]
        nc = [torch.tensor([c[1], c[0]]), torch.tensor([c[2], c[1]]), torch.tensor([c[0], c[2]])]
        for j in range(i+1, len(triangle_set)):
            a = triangle_set[j]
            pa = [torch.tensor([a[0], a[1]]), torch.tensor([a[1], a[2]]), torch.tensor([a[2], a[0]])]
            na = [torch.tensor([a[1], a[0]]), torch.tensor([a[2], a[1]]), torch.tensor([a[0], a[2]])]
            for x in pc:
                for y in pa:
                    if torch.equal(x, y):
                        d1d1s[i, j] = 1
                        d1d1s[j, i] = 1
                        break
                for y in na:
                    if torch.equal(x, y):
                        d1d1s[i, j] = -1
                        d1d1s[j, i] = -1
                        break
    return d1d1s


def curl(triangle_set, edge_index, edge_attr, Nm=None):
    curl_ls = []
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=edge_attr,
                       sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    adj = adj - adj.permute(1, 0)

    for tri in triangle_set:
        accumulate_curl = 0
        accumulate_curl += adj[tri[0], tri[1]]
        accumulate_curl += adj[tri[1], tri[2]]
        accumulate_curl += adj[tri[2], tri[0]]
        curl_ls.append(accumulate_curl)
    return torch.tensor(curl_ls)


def delta_1_star(gamma, triangle_set, edge_index, Nm=None):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    adj = torch.zeros(Nm, Nm)
    for i in range(len(triangle_set)):
        c = triangle_set[i]
        adj[c[0], c[1]] += gamma[i]
        adj[c[1], c[0]] -= gamma[i]
        adj[c[1], c[2]] += gamma[i]
        adj[c[2], c[1]] -= gamma[i]
        adj[c[2], c[0]] += gamma[i]
        adj[c[0], c[2]] -= gamma[i]
    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    return A * adj



def compute_solenoidal(edge_index, edge_attr, Nm=None, directed=True):
    triangle_set = extract_triangle(edge_index, Nm, directed)
    d1d1s = delta_1_delta_1_star(triangle_set)
    curls = curl(triangle_set, edge_index, edge_attr, Nm)
    gamma = torch.linalg.solve(d1d1s, curls)
    w_s = delta_1_star(gamma, triangle_set, edge_index, Nm)
    return w_s


def Hodge_decomposition(edge_index, edge_attr, Nm=None, directed=True):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=edge_attr,
                       sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    w_g = compute_gradient(edge_index, edge_attr, Nm)
    w_s = compute_solenoidal(edge_index, edge_attr, Nm, directed)
    w_h = adj - w_s - w_g
    return w_g, w_s, w_h


def compute_Hodge_0_Laplacian(edge_index, Nm=None, directed=True):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    if not directed:
        A = SparseTensor(row=edge_index[0],
                         col=edge_index[1],
                         value=torch.ones(edge_index.shape[1], dtype=torch.float),
                         sparse_sizes=(Nm, Nm)).coalesce().to_dense()
        # convert to standard directed graph
        # this won't affect result for directed graph
        # but should be used for undirected graphs
        A = (A + A.permute(1, 0)).bool().float()
        mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
        A = A * mask
        edge_index, _ = dense_to_sparse(A)

    B1 = torch.zeros([Nm, edge_index.shape[1]], dtype=torch.float)
    for i in range(edge_index.shape[1]):
        B1[edge_index[0, i], i] = -1
        B1[edge_index[1, i], i] = 1

    return torch.matmul(B1, B1.permute(1, 0))


def compute_Helmholtzians_Hodge_1_Laplacian(edge_index, Nm=None, directed=True):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    A = SparseTensor(row=edge_index[0],
                     col=edge_index[1],
                     value=torch.ones(edge_index.shape[1], dtype=torch.float),
                     sparse_sizes=(Nm, Nm)).coalesce().to_dense()
    # convert to standard directed graph
    # this won't affect result for directed graph
    # but should be used for undirected graphs, and convenient for computing B2
    A = (A + A.permute(1, 0)).bool().float()
    mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
    A = A * mask
    edge_index, _ = dense_to_sparse(A)
    # print(edge_index)

    B1 = torch.zeros([Nm, edge_index.shape[1]], dtype=torch.float)
    for i in range(edge_index.shape[1]):
        B1[edge_index[0, i], i] = -1
        B1[edge_index[1, i], i] = 1

    triangle_set = extract_triangle(edge_index, Nm, True)
    B2 = torch.zeros([edge_index.shape[1], len(triangle_set)])
    for i in range(len(triangle_set)):
        c = triangle_set[i]
        for j in range(edge_index.shape[1]):
            if edge_index[0, j] == c[0] and edge_index[1, j] == c[1]:
                B2[j, i] = 1
                break
        for j in range(edge_index.shape[1]):
            if edge_index[0, j] == c[1] and edge_index[1, j] == c[2]:
                B2[j, i] = 1
                break
        for j in range(edge_index.shape[1]):
            if edge_index[0, j] == c[0] and edge_index[1, j] == c[2]:
                B2[j, i] = -1
                break
    return torch.matmul(B1.permute(1, 0), B1) + torch.matmul(B2, B2.permute(1, 0))


def display_L1_eigen(index, edge_index, Nm=None, directed=True):
    if Nm is None:
        Nm = torch.max(edge_index) + 1
    G = nx.Graph()
    G.clear()
    G.add_nodes_from(np.arange(0, Nm))
    edge_list = []
    for j in range(edge_index.shape[1]):
        # print((data.edge_index[0][j], data.edge_index[1][j]))
        edge_list.append((edge_index[0][j].item(), edge_index[1][j].item()))
    # print(edge_list)
    G.add_edges_from(edge_list)

    L1 = compute_Helmholtzians_Hodge_1_Laplacian(edge_index, Nm, directed)
    eigenvalue, eigenvector = torch.linalg.eig(L1)
    eigenvector = torch.view_as_real(eigenvector)[:, :, 0]
    eigenvalue = torch.view_as_real(eigenvalue)[:, 0]

    # cmap = plt.cm.get_cmap('Blues')
    cmap = matplotlib.colormaps['Blues']
    edge_vmin, edge_vmax = 0, torch.max(eigenvector).item() * 0.9
    print(eigenvalue, eigenvector)
    for i in range(edge_index.shape[1]):
        print(eigenvalue[i], eigenvector[:, i])
        plt.figure(i, figsize=(20, 20))
        nx.draw_networkx(G, node_size=300, edge_color=np.abs(eigenvector[:, i].numpy()), width=5.0, edge_cmap=cmap, edge_vmin=edge_vmin, edge_vmax=edge_vmax)
        plt.title(str(index) + '_' + str(eigenvalue[i].item()))
        plt.savefig('graph_figure/' + str(index) + '_' + str(eigenvalue[i].item()) + '.png')


if __name__ == '__main__':
    # edge_index = torch.tensor([[1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7, 2, 8, 3, 6, 4, 5, 6, 8, 5, 6, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
    #                            [2, 8, 3, 6, 4, 5, 6, 8, 5, 6, 6, 7, 8, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]])
    # edge_index = edge_index - 1
    # edge_index[1, :26] += 8
    # edge_index[0, 26:] += 8
    # # edge_attr = torch.tensor([-1, 4.2, -2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9, 4.1, -1, -2, -1, 4.2, -2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9,
    # #      4.1, -1, -2, 3.1, 5.1, 23.9, 13.1, 13.1, 27.9, -3, 9.3])
    # edge_attr = torch.tensor([1, 4.2, 2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9, 4.1, 1, 2, 1, 4.2, 2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9, 4.1,
    #  1, 2, 5.1, 11.1, 27.9, 13.1, 13.1, 29.9, 3, 13.3])
    #
    # Nm = 16

    # example in Tutorial
    # edge_index = torch.tensor([[1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7],
    #                            [2, 8, 3, 6, 4, 5, 6, 8, 5, 6, 6, 7, 8]])
    # edge_index = edge_index - 1
    # edge_attr = torch.tensor(
    #     [-1, 4.2, -2, 8.1, 3.1, 5.9, 9.8, 7.1, 3.1, 6.9, 4.1, -1, -2])
    # Nm = 8

    # example in Hodge Laplacian
    # edge_index = torch.tensor([[1, 2, 3, 3, 3, 4, 5],
    #                            [2, 3, 4, 5, 6, 1, 6]])
    # edge_index -= 1
    # Nm = 6
    # edge_index = torch.tensor([[1, 2, 3, 3, 4, 4, 6],
    #                            [2, 3, 4, 5, 1, 6, 2]])
    # edge_index -= 1
    # Nm = 6

    # edge_index = torch.tensor([[0, 1, 2],
    #                            [1, 2, 0]])
    # edge_attr = torch.tensor([1., 1.1, 0.9])
    # Nm = 3

    # example zinc molecular graph 1002
    # index = 1002
    # edge_index = torch.tensor([[0, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 11, 13, 13, 14, 15, 16, 17, 19],
    #                            [1, 2, 3, 20,4, 5, 6, 19,7, 10,8, 9, 10,11, 12, 13, 14, 18, 15, 16, 17, 18, 20]])
    # Nm = 21

    # example zinc molecular graph 798
    # index = 798
    # edge_index = torch.tensor([[0, 1, 1,  2, 3, 3, 5, 6, 6, 7, 8, 9, 9, 11, 11, 12, 12, 16, 16, 17, 18, 19, 19, 20, 21, 22, 23],
    #                            [1, 2, 16, 3, 4, 5, 6, 7, 15,8, 9, 10,11,12, 15, 13, 14, 17, 24, 18, 19, 20, 24, 21, 22, 23, 24]])
    # Nm = 25

    # example zinc molecular graph 488
    index = 488
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 4, 5, 6, 6, 8, 9, 9, 10, 11, 11],
                               [1, 2, 3, 4, 5, 14,6, 7, 8, 9, 10,14,11, 12, 13]])
    Nm = 15
    # w_g, w_s, w_h = Hodge_decomposition(edge_index, edge_attr, Nm, True)
    # print(w_g[:8, 8:])
    # print(w_s[:8, 8:])
    # print(w_h[:8, 8:])
    # print((w_g + w_h + w_s)[:8, 8:])
    # L0 = compute_graph_Laplacian_0(edge_index, Nm)
    # print(L0)
    # L0 = compute_Hodge_0_Laplacian(edge_index, Nm)
    # print(L0)
    # L1 = compute_Helmholtzians_Hodge_1_Laplacian(edge_index, Nm, True)
    # print(L1)
    # eigenvalue, eigenvector = torch.linalg.eig(L1)
    # eigenvector = torch.view_as_real(eigenvector)[:, :, 0]
    # print(eigenvalue, eigenvector)
    # for i in range(15):
    #     print(eigenvalue[i], eigenvector[:, i])
    display_L1_eigen(index, edge_index, Nm, True)

