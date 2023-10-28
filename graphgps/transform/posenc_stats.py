from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
import networkx as nx
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, dense_to_sparse, coalesce, k_hop_subgraph)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from graphgps.transform.hodge_decomposition import *
from torch_sparse import SparseTensor
from functools import partial
from .rrwp import add_full_rrwp
# from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse, coalesce


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.
    'RRWP': Relative Random Walk Probabilities PE (Ours, for GRIT)

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE',
                     'HodgeLap1PE', 'EdgeRWSE', 'RRWP', 'InterRWSE', 'RD']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower()
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())
        
        if 'LapPE' in pe_types:
            max_freqs=cfg.posenc_LapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm
        elif 'EquivStableLapPE' in pe_types:  
            max_freqs=cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_EquivStableLapPE.eigen.eigvec_norm
        
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)

    if 'SignNet' in pe_types:
        # Eigen-decomposition with numpy for SignNet.
        norm_type = cfg.posenc_SignNet.eigen.laplacian_norm.lower()
        if norm_type == 'none':
            norm_type = None
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=norm_type,
                           num_nodes=N)
        )
        evals_sn, evects_sn = np.linalg.eigh(L.toarray())
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn, evects=evects_sn,
            max_freqs=cfg.posenc_SignNet.eigen.max_freqs,
            eigvec_norm=cfg.posenc_SignNet.eigen.eigvec_norm)

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing

    # Heat Kernels.
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        if laplacian_norm_type is not None or evals is None or evects is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = evals, evects
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        # Get the full heat kernels.
        if 'HKfullPE' in pe_types:
            # The heat kernels can't be stored in the Data object without
            # additional padding because in PyG's collation of the graphs the
            # sizes of tensors must match except in dimension 0. Do this when
            # the full heat kernels are actually used downstream by an Encoder.
            raise NotImplementedError()
            # heat_kernels, hk_diag = get_heat_kernels(evects_heat, evals_heat,
            #                                   kernel_times=kernel_param.times)
            # data.pestat_HKdiagSE = hk_diag
        # Get heat kernel diagonals in more efficient way.
        if 'HKdiagSE' in pe_types:
            kernel_param = cfg.posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param.times,
                                            space_dim=0)
            data.pestat_HKdiagSE = hk_diag

    # Electrostatic interaction inspired kernel.
    if 'ElstaticSE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticSE = elstatic

    if 'HodgeLap1PE' in pe_types:
        Nm = data.num_nodes
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=torch.ones(data.edge_index.shape[1], dtype=torch.float),
                         sparse_sizes=(Nm, Nm)).coalesce().to_dense()
        A = (A + A.permute(1, 0)).bool().float()
        deg = A.sum(dim=1)
        data.log_deg = torch.log(deg + 1)
        data.deg = deg.type(torch.long)
        mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
        A = A * mask
        directed_edge_index, _ = dense_to_sparse(A)
        L1 = compute_Helmholtzians_Hodge_1_Laplacian(undir_edge_index, N, False)
        eigenvalue, eigenvector = torch.linalg.eigh(L1)
        # eigenvector = torch.view_as_real(eigenvector)[:, :, 0]
        # eigenvalue = torch.view_as_real(eigenvalue)[:, 0]
        max_zero_freq = cfg.posenc_HodgeLap1PE.max_zero_freq
        embed_type = cfg.posenc_HodgeLap1PE.embed_type
        zero_vec = []
        low_vec = []
        all_vec = []
        if embed_type == 'sum_zero_abs':
            final_dim = 1
            for i in range(len(eigenvalue)):
                if torch.abs(eigenvalue[i]) < 1e-4:
                    zero_vec.append(torch.abs(eigenvector[:, i]).unsqueeze(0))
            # print(len(zero_vec))
            if len(zero_vec) > 0:
                zero_vec = torch.cat(zero_vec, dim=0)
                zero_vec = torch.sum(zero_vec, dim=0).unsqueeze(1)
            else:
                zero_vec = torch.zeros([int(undir_edge_index.shape[1]/2), 1], dtype=torch.float)
            posenc_vec = zero_vec
        elif embed_type == 'proj_zero':
            final_dim = max_zero_freq
            for i in range(len(eigenvalue)):
                if torch.abs(eigenvalue[i]) < 1e-4:
                    zero_vec.append(eigenvector[:, i].unsqueeze(0))
                if len(zero_vec) == max_zero_freq:
                    break
            if len(zero_vec) > 0:
                zero_vec = torch.cat(zero_vec, dim=0)
                zero_vec = zero_vec.permute(1, 0)
                if max_zero_freq > zero_vec.shape[1]:
                    zero_vec = torch.cat([zero_vec, torch.zeros([zero_vec.shape[0], max_zero_freq-zero_vec.shape[1]], dtype=torch.float)], dim=1)
            else:
                zero_vec = torch.zeros([int(undir_edge_index.shape[1]/2), max_zero_freq], dtype=torch.float)
            posenc_vec = zero_vec
        elif embed_type == 'proj_low':
            m = len(eigenvalue)
            num_low_freq = cfg.posenc_HodgeLap1PE.max_low_freq
            num_total_freq = cfg.posenc_HodgeLap1PE.max_total_freq
            final_dim = 2 + 2 * num_total_freq
            values, indices = torch.sort(eigenvalue, descending=False)
            for i in range(min(m, num_total_freq)):
                vec = eigenvector[:, indices[i]].unsqueeze(0)
                if values[i] < 1e-4 and len(zero_vec) < max_zero_freq:
                    zero_vec.append(vec)
                elif values[i] > 1e-4 and len(low_vec) < num_low_freq:
                    low_vec.append(vec)
                catted = torch.cat([vec, torch.ones([1, m]) * values[i]])
                all_vec.append(catted)
            if len(all_vec) > 0:
                all_vec = torch.cat(all_vec).permute(1, 0)  # [m, 2*num_total_freq]
            else:
                all_vec = torch.zeros([m, 2 * num_total_freq], dtype=torch.float)
            if 2 * num_total_freq > all_vec.shape[1]:
                all_vec = torch.cat([all_vec, torch.zeros([m, 2 * num_total_freq - all_vec.shape[1]], dtype=torch.float)], dim=1)
            if len(zero_vec) > 0:
                zero_vec = torch.cat(zero_vec, dim=0)
                proj_zero = torch.matmul(zero_vec.permute(1, 0), zero_vec)
                unit = torch.ones([m, 1], dtype=torch.float) #/ torch.sqrt(torch.tensor(m, dtype=torch.float))
                zero_vec = torch.matmul(torch.abs(proj_zero), unit)
                # if max_zero_freq > zero_vec.shape[1]:
                #     zero_vec = torch.cat([zero_vec, torch.zeros([zero_vec.shape[0], max_zero_freq-zero_vec.shape[1]], dtype=torch.float)], dim=1)
            else:
                zero_vec = torch.zeros([m, 1], dtype=torch.float)
            if len(low_vec) > 0:
                low_vec = torch.cat(low_vec, dim=0)
                proj_low = torch.matmul(low_vec.permute(1, 0), low_vec)
                unit = torch.ones([m, 1], dtype=torch.float) / torch.sqrt(torch.tensor(m, dtype=torch.float))
                low_vec = torch.matmul(torch.abs(proj_low), unit)
                # if num_low_freq > low_vec.shape[1]:
                #     low_vec = torch.cat([low_vec, torch.zeros([low_vec.shape[0], num_low_freq-low_vec.shape[1]], dtype=torch.float)], dim=1)
            else:
                low_vec = torch.zeros([m, 1], dtype=torch.float)
            posenc_vec = torch.cat([zero_vec, low_vec, all_vec], dim=1)
        else:
            raise NotImplementedError
        undir_edge_index_, vec_undirected = coalesce(
            torch.cat([directed_edge_index,
                       torch.cat([directed_edge_index[1].unsqueeze(0), directed_edge_index[0].unsqueeze(0)], dim=0)],
                      dim=1),
            torch.cat([posenc_vec, posenc_vec], dim=0)
        )
        # vec_undirected = torch.zeros([undir_edge_index.shape[1], final_dim], dtype=torch.float)
        # idx = 0
        # for i in range(undir_edge_index.shape[1]):
        #     if undir_edge_index[0, i] < undir_edge_index[1, i]:
        #         vec_undirected[i] = posenc_vec[idx]
        #         idx += 1
        #     else:
        #         for j in range(i):
        #             if undir_edge_index[0, j] == undir_edge_index[1, i] and undir_edge_index[1, j] == undir_edge_index[0, i]:
        #                 vec_undirected[i] = vec_undirected[j]
        #                 break
        data.pestat_HodgeLap1PE = vec_undirected
        data.num_undir_edges = torch.tensor([undir_edge_index.shape[1]])

    if 'EdgeRWSE' in pe_types:
        directed_walk = cfg.posenc_EdgeRWSE.directed_walk
        internal_directed = cfg.posenc_EdgeRWSE.get('internal_directed', False)
        walk_time = len(cfg.posenc_EdgeRWSE.kernel.times)
        Nm = data.num_nodes
        if not cfg.posenc_EdgeRWSE.local:
            if not internal_directed:
                A = SparseTensor(row=data.edge_index[0],
                                 col=data.edge_index[1],
                                 value=torch.ones(data.edge_index.shape[1], dtype=torch.float),
                                 sparse_sizes=(Nm, Nm)).coalesce().to_dense()
                A = (A + A.permute(1, 0)).bool().float()
                deg = A.sum(dim=1)
                data.log_deg = torch.log(deg + 1)
                data.deg = deg.type(torch.long)
                mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
                A = A * mask
                directed_edge_index, _ = dense_to_sparse(A)
                if not directed_walk:
                    B1 = torch.zeros([Nm, directed_edge_index.shape[1]], dtype=torch.float)
                    for i in range(directed_edge_index.shape[1]):
                        B1[directed_edge_index[0, i], i] = 1
                        B1[directed_edge_index[1, i], i] = 1

                    B = torch.matmul(B1.permute(1, 0), B1)
                    B = B - torch.eye(directed_edge_index.shape[1]) * 2.
                    # B = torch.matmul(torch.diag_embed(1. / torch.sum(B, dim=0)), B)
                    for j in range(B.shape[0]):
                        if torch.sum(B[j]) > 0:
                            B[j] = B[j] / torch.sum(B[j])
                    Bk = torch.eye(directed_edge_index.shape[1])
                    prob = []
                    for walks in range(walk_time):
                        Bk = torch.matmul(B, Bk)
                        prob.append(torch.diagonal(Bk).unsqueeze(0))
                    prob = torch.cat(prob, dim=0).permute(1, 0)

                else:
                    B1 = torch.zeros([Nm, directed_edge_index.shape[1]], dtype=torch.float)
                    for i in range(directed_edge_index.shape[1]):
                        B1[directed_edge_index[0, i], i] = 1
                        B1[directed_edge_index[1, i], i] = 1
                    delta_1 = B1 / 2
                    delta_1_s = B1
                    for i in range(Nm):
                        if int(torch.sum(B1[i])) <= 1:
                            pass
                        else:
                            delta_1_s[i] = delta_1_s[i] / (torch.sum(delta_1_s[i]) - 1)
                    B = torch.matmul(delta_1.permute(1, 0), delta_1_s)
                    B = B - torch.diag_embed(torch.diagonal(B))
                    # B = torch.matmul(torch.diag_embed(1 / torch.sum(B, dim=0)), B)
                    for j in range(B.shape[0]):
                        if torch.sum(B[j]) > 0:
                            B[j] = B[j] / torch.sum(B[j])
                    Bk = torch.eye(directed_edge_index.shape[1])
                    prob = []
                    for walks in range(walk_time):
                        Bk = torch.matmul(B, Bk)
                        prob.append(torch.diagonal(Bk).unsqueeze(0))
                    prob = torch.cat(prob, dim=0).permute(1, 0)

                undir_edge_index_, prob_undirected = coalesce(
                    torch.cat([directed_edge_index, torch.cat([directed_edge_index[1].unsqueeze(0), directed_edge_index[0].unsqueeze(0)], dim=0)], dim=1),
                    torch.cat([prob, prob], dim=0)
                )
                # idx = 0
                # prob_undirected = torch.zeros([undir_edge_index.shape[1], walk_time], dtype=torch.float)
                # for i in range(undir_edge_index.shape[1]):
                #     if undir_edge_index[0, i] < undir_edge_index[1, i]:
                #         prob_undirected[i] = prob[idx]
                #         idx += 1
                #     else:
                #         for j in range(i):
                #             if undir_edge_index[0, j] == undir_edge_index[1, i] and undir_edge_index[1, j] == \
                #                     undir_edge_index[0, i]:
                #                 prob_undirected[i] = prob_undirected[j]
                #                 break
                data.pestat_EdgeRWSE = prob_undirected
            else:
                B1 = torch.zeros([data.edge_index.shape[1], Nm], dtype=torch.float)
                B2 = torch.zeros([Nm, data.edge_index.shape[1]], dtype=torch.float)
                for i in range(data.edge_index.shape[1]):
                    B1[i, data.edge_index[1, i]] = 1
                    B2[data.edge_index[0, i], i] = 1
                P = torch.matmul(B1, B2)
                # print(P)
                for j in range(P.shape[0]):
                    if torch.sum(P[j]) > 0:
                        P[j] = P[j] / torch.sum(P[j])

                prob = []
                Bk = torch.eye(data.edge_index.shape[1])
                for walks in range(walk_time):
                    Bk = torch.matmul(Bk, P)
                    prob.append(torch.diagonal(Bk).unsqueeze(0))
                prob = torch.cat(prob, dim=0).permute(1, 0)
                data.pestat_EdgeRWSE = prob
        else:
            A = SparseTensor(row=data.edge_index[0],
                             col=data.edge_index[1],
                             value=torch.ones(data.edge_index.shape[1], dtype=torch.float),
                             sparse_sizes=(Nm, Nm)).coalesce().to_dense()
            A = (A + A.permute(1, 0)).bool().float()
            mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
            A = A * mask
            full_directed_edge_index, _ = dense_to_sparse(A)
            # prob_undir = torch.zeros([undir_edge_index.shape[1], walk_time], dtype=torch.float)
            probs = []
            for i in range(undir_edge_index.shape[1]):
                if i % 100 == 0:
                    print(i)
                if undir_edge_index[0, i] < undir_edge_index[1, i]:
                    (subset, subg_edge_index, mapping, edge_mask) = k_hop_subgraph(undir_edge_index[:, i], cfg.posenc_EdgeRWSE.local_hop, undir_edge_index)
                    idx = 0
                    for j in range(subg_edge_index.shape[-1]):
                        if torch.equal(subg_edge_index[:, j], undir_edge_index[:, 6]):
                            break
                        if subg_edge_index[0, j] < subg_edge_index[1, j]:
                            idx = idx + 1

                    (subset, subg_edge_index, mapping, edge_mask) = k_hop_subgraph(undir_edge_index[:, 6], cfg.posenc_EdgeRWSE.local_hop, undir_edge_index,
                                                                                   relabel_nodes=True)
                    adj = SparseTensor(row=subg_edge_index[0],
                                       col=subg_edge_index[1],
                                       value=torch.ones(subg_edge_index.shape[1], dtype=torch.float),
                                       sparse_sizes=(subset.shape[-1], subset.shape[-1])).coalesce().to_dense()
                    mask = torch.triu(torch.ones(subset.shape[-1], subset.shape[-1]), diagonal=1)
                    adj = adj * mask
                    # print(adj)
                    directed_edge_index = dense_to_sparse(adj)[0]
                    B1 = torch.zeros([subset.shape[-1], directed_edge_index.shape[1]], dtype=torch.float)
                    for i in range(directed_edge_index.shape[1]):
                        B1[directed_edge_index[0, i], i] = 1
                        B1[directed_edge_index[1, i], i] = 1

                    B = torch.matmul(B1.permute(1, 0), B1)
                    B = B - torch.eye(directed_edge_index.shape[1]) * 2.
                    # B = torch.matmul(torch.diag_embed(1. / torch.sum(B, dim=0)), B)
                    for j in range(B.shape[0]):
                        if torch.sum(B[j]) > 0:
                            B[j] = B[j] / torch.sum(B[j])
                    Bk = torch.eye(directed_edge_index.shape[1])
                    prob = []
                    for walks in range(5):
                        Bk = torch.matmul(B, Bk)
                        prob.append(Bk[idx, idx].unsqueeze(0))
                    prob = torch.cat(prob, dim=0)
                    probs.append(prob.unsqueeze(0))
            probs = torch.cat(probs, dim=0)
            undir_edge_index_, prob_undirected = coalesce(
                torch.cat([full_directed_edge_index,
                           torch.cat([full_directed_edge_index[1].unsqueeze(0), full_directed_edge_index[0].unsqueeze(0)],
                                     dim=0)], dim=1),
                torch.cat([probs, probs], dim=0)
            )
            data.pestat_EdgeRWSE = prob_undirected


    if 'RRWP' in pe_types:
        param = cfg.posenc_RRWP
        transform = partial(add_full_rrwp,
                            walk_length=param.ksteps,
                            attr_name_abs="rrwp",
                            attr_name_rel="rrwp",
                            add_identity=True,
                            spd=param.spd,  # by default False
                            )
        data = transform(data)

    if 'RD' in pe_types:
        N = data.num_nodes
        adj = np.zeros((N, N), dtype=np.float32)
        adj[undir_edge_index[0, :], undir_edge_index[1, :]] = 1.0

        deg = torch.tensor(adj, dtype=torch.float).sum(dim=1)
        data.log_deg = torch.log(deg + 1)
        data.deg = deg.type(torch.long)

        # 2) connected_components
        g = nx.Graph(adj)
        g_components_list = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        g_resistance_matrix = np.zeros((N, N)) - 1.0
        g_index = 0
        for item in g_components_list:
            cur_adj = nx.to_numpy_array(item)
            cur_num_nodes = cur_adj.shape[0]
            cur_res_dis = np.linalg.pinv(
                np.diag(cur_adj.sum(axis=-1)) - cur_adj + np.ones((cur_num_nodes, cur_num_nodes),
                                                                  dtype=np.float32) / cur_num_nodes
            ).astype(np.float32)
            A = np.diag(cur_res_dis)[:, None]
            B = np.diag(cur_res_dis)[None, :]
            cur_res_dis = A + B - 2 * cur_res_dis
            g_resistance_matrix[g_index:g_index + cur_num_nodes, g_index:g_index + cur_num_nodes] = cur_res_dis
            g_index += cur_num_nodes
        g_cur_index = []
        for item in g_components_list:
            g_cur_index.extend(list(item.nodes))
        ori_idx = np.arange(N)
        g_resistance_matrix[g_cur_index, :] = g_resistance_matrix[ori_idx, :]
        g_resistance_matrix[:, g_cur_index] = g_resistance_matrix[:, ori_idx]

        if g_resistance_matrix.max() > N - 1:
            print(f'error: {g_resistance_matrix}')
        g_resistance_matrix[g_resistance_matrix == -1.0] = 512.0
        g_resistance_matrix = torch.tensor(g_resistance_matrix, dtype=torch.float)

        RD_index, RD_val = dense_to_sparse(g_resistance_matrix)
        data.RD_index = RD_index
        data.RD_val = RD_val.reshape(-1, 1)

    if 'InterRWSE' in pe_types:
        walk_time = len(cfg.posenc_InterRWSE.kernel.times)
        Nm = data.num_nodes
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=torch.ones(data.edge_index.shape[1], dtype=torch.float),
                         sparse_sizes=(Nm, Nm)).coalesce().to_dense()
        A = (A + A.permute(1, 0)).bool().float()
        mask = torch.triu(torch.ones(Nm, Nm), diagonal=1)
        A_mask = A * mask
        directed_edge_index, _ = dense_to_sparse(A_mask)
        B1 = torch.zeros([Nm, directed_edge_index.shape[1]], dtype=torch.float)
        for i in range(directed_edge_index.shape[1]):
            B1[directed_edge_index[0, i], i] = 1
            B1[directed_edge_index[1, i], i] = 1
        B = torch.matmul(B1.permute(1, 0), B1)
        B = B - torch.eye(directed_edge_index.shape[1]) * 2.
        # B = torch.matmul(torch.diag_embed(1. / torch.sum(B, dim=0)), B)
        for j in range(B.shape[0]):
            if torch.sum(B[j]) > 0:
                B[j] = B[j] / torch.sum(B[j])
        P_u = torch.cat([A, B1], dim=1)
        for j in range(P_u.shape[0]):
            if torch.sum(P_u[j]) > 0:
                P_u[j] = (P_u[j] / torch.sum(P_u[j]))
        P_d = torch.cat([B1.permute(1, 0) / 4., B / 2.], dim=1)
        P = torch.cat([P_u, P_d], dim=0)
        # for j in range(P.shape[0]):
        #     if torch.sum(P[j]) > 0:
        #         P[j] = P[j] / torch.sum(P[j])
        Bk = torch.eye(directed_edge_index.shape[1] + data.num_nodes)
        prob = []
        for walks in range(walk_time):
            Bk = torch.matmul(P, Bk)
            prob.append(torch.diagonal(Bk).unsqueeze(0))
        prob = torch.cat(prob, dim=0).permute(1, 0)

        data.pestat_RWSE = prob[:data.num_nodes]
        prob_edge = prob[data.num_nodes:]

        undir_edge_index_, prob_undirected = coalesce(
            torch.cat([directed_edge_index,
                       torch.cat([directed_edge_index[1].unsqueeze(0), directed_edge_index[0].unsqueeze(0)], dim=0)],
                      dim=1),
            torch.cat([prob_edge, prob_edge], dim=0)
        )

        # idx = 0
        # prob_undirected = torch.zeros([undir_edge_index.shape[1], walk_time], dtype=torch.float)
        # for i in range(undir_edge_index.shape[1]):
        #     if undir_edge_index[0, i] < undir_edge_index[1, i]:
        #         prob_undirected[i] = prob_edge[idx]
        #         idx += 1
        #     else:
        #         for j in range(i):
        #             if undir_edge_index[0, j] == undir_edge_index[1, i] and undir_edge_index[1, j] == \
        #                     undir_edge_index[0, i]:
        #                 prob_undirected[i] = prob_undirected[j]
        #                 break
        data.pestat_EdgeRWSE = prob_undirected

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.max(dim=0)[0],  # Max of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.max(dim=0)[0],  # Max of Vj -> i
        electrostatic.mean(dim=1),  # Mean of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData

class ComputePosencStat(BaseTransform):
    def __init__(self, pe_types, is_undirected, cfg):
        self.pe_types = pe_types
        self.is_undirected = is_undirected
        self.cfg = cfg

    def __call__(self, data: Data) -> Data:
        data = compute_posenc_stats(data, pe_types=self.pe_types,
                                    is_undirected=self.is_undirected,
                                    cfg=self.cfg
                                    )
        return data
