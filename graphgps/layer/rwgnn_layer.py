import random
from codecs import ascii_encode
from typing import Final, List, Tuple, Optional
# from util import cumsum_pad0, deg2rowptr, extracttuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.nn as pygnn
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse, coalesce
from torch_sparse import SparseTensor
from tkinter import _flatten
from math import factorial
import math


# class RW_MPNN_layer(nn.Module):
#
#     def __init__(self,
#                  nn_layer: nn.ModuleList,
#                  num_layer_MPNN: int = 1,
#                  similarity_type: str = 'cos',
#                  inference_mode: str = 'original',
#                  mp_threshold: float = 0.0,
#                  force_undirected: bool = False):
#         super().__init__()
#
#         self.GINE = nn.ModuleList([pygnn.GINEConv(nn_layer[_]) for _ in range(num_layer_MPNN)])
#         self.num_layer_MPNN = num_layer_MPNN
#         self.similarity_type = similarity_type
#         self.inference_mode = inference_mode
#         self.mp_threshold = mp_threshold
#         self.force_undirected = force_undirected
#
#     def forward(self, x, edge_index, edge_attr, batch, max_num_nodes=None) -> Tensor:
#         # B, Nm = adj.shape[0], adj.shape[1]
#         # Bidx = torch.arange(B, device=x.device)
#         # null_node_mask = torch.zeros((B, Nm + 1), device=x.device)
#         # null_node_mask[Bidx, num_node] = 1
#         # null_node_mask = null_node_mask.cumsum_(dim=1) > 0
#         # null_node_mask = null_node_mask[:, :-1]
#         # real_node_mask = ~null_node_mask
#         #
#         # adj_original = adj
#         # x_padded = torch.zeros(B, Nm, x.shape[-1])
#         # x_padded[real_node_mask] = x
#
#         # note that x is not dense originally
#         edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='add')
#         adj = to_dense_adj(edge_index, batch, max_num_nodes=max_num_nodes)
#         B, Nm = adj.shape[0], adj.shape[1]
#
#         for _ in range(self.num_layer_MPNN):
#             # Random Walk MPNN
#             x_padded, real_node_mask = to_dense_batch(x, batch)
#             if self.similarity_type == 'cos':
#                 similarity = torch.cosine_similarity(x_padded.unsqueeze(2), x_padded.unsqueeze(1), dim=-1)
#             # elif self.similarity_type == 'project_edge':
#             #     pro1 = torch.einsum("bijd,bjd->bijd", adj, x)
#             #     pro2 = torch.einsum("bijd,bid->bijd", adj, x)
#             #     similarity = torch.cosine_similarity(pro1, pro2, dim=-1)
#             else:
#                 raise NotImplementedError
#
#             if self.inference_mode == 'sample':  # MCMC, stochastic sampling method
#                 # TODO: strategy gradient; layer level (equivalent to GAT, probability as weight and mean effect)
#                 #  or model level sampling? replaced by variational inference
#                 rdn = torch.rand(B, Nm, Nm).to(x.device)
#                 if self.force_undirected:
#                     rdn = (rdn + rdn.permute(0, 2, 1)) / 2
#                 sampled = torch.le(rdn, (similarity + 1.) / 2.).float()  # 之后考虑策略梯度
#                 rw_adj = torch.einsum("bij,bij->bij", adj, sampled)
#             elif self.inference_mode == 'average':  # Deterministic algorithm
#                 sampled = torch.ge((similarity + 1.) / 2., self.mp_threshold).float()
#                 rw_adj = torch.einsum("bij,bij->bij", adj, sampled)
#             elif self.inference_mode == 'original':
#                 rw_adj = adj
#             else:
#                 raise NotImplementedError
#             edge_mask = rw_adj[adj.gt(0)] == adj[adj.gt(0)]
#             x = self.GINE[_](x, edge_index[:, edge_mask], edge_attr[edge_mask])
#
#         return x


class RW_MPNN_layer(nn.Module):

    def __init__(self,
                 nn_layer: nn.ModuleList,
                 nn_layer_2: nn.ModuleList,
                 num_layer_MPNN: int = 1,
                 similarity_type: str = 'cos',
                 inference_mode: str = 'original',
                 mp_threshold: float = 0.0,
                 force_undirected: bool = False):
        super().__init__()

        self.GINE = nn.ModuleList([pygnn.GINEConv(nn_layer[_]) for _ in range(num_layer_MPNN)])
        self.GINE_intra = nn.ModuleList([pygnn.GINEConv(nn_layer_2[_]) for _ in range(num_layer_MPNN)])
        self.num_layer_MPNN = num_layer_MPNN
        self.similarity_type = similarity_type
        self.inference_mode = inference_mode
        self.mp_threshold = mp_threshold
        self.force_undirected = force_undirected

    def forward(self, x, edge_index, edge_attr) -> Tensor:
        # B, Nm = adj.shape[0], adj.shape[1]
        # Bidx = torch.arange(B, device=x.device)
        # null_node_mask = torch.zeros((B, Nm + 1), device=x.device)
        # null_node_mask[Bidx, num_node] = 1
        # null_node_mask = null_node_mask.cumsum_(dim=1) > 0
        # null_node_mask = null_node_mask[:, :-1]
        # real_node_mask = ~null_node_mask
        #
        # adj_original = adj
        # x_padded = torch.zeros(B, Nm, x.shape[-1])
        # x_padded[real_node_mask] = x

        # note that x is not dense originally

        for _ in range(self.num_layer_MPNN):
            # Random Walk MPNN
            if self.similarity_type == 'cos':
                x_src = x[edge_index[0]]
                x_des = x[edge_index[1]]
                similarity = torch.cosine_similarity(x_src, x_des, dim=-1)
            # elif self.similarity_type == 'project_edge':
            #     pro1 = torch.einsum("bijd,bjd->bijd", adj, x)
            #     pro2 = torch.einsum("bijd,bid->bijd", adj, x)
            #     similarity = torch.cosine_similarity(pro1, pro2, dim=-1)
            else:
                raise NotImplementedError

            if self.inference_mode == 'sample' or self.inference_mode == 'inter_intra':  # MCMC, stochastic sampling method
                # TODO: strategy gradient; layer level (equivalent to GAT, probability as weight and mean effect)
                #  or model level sampling? replaced by variational inference
                rdn = torch.rand(similarity.shape).to(x.device)
                # if self.force_undirected:
                #     rdn = (rdn + rdn.permute(0, 2, 1)) / 2
                edge_mask = torch.le(rdn, (similarity + 1.) / 2.)  # 之后考虑策略梯度
            elif self.inference_mode == 'average':  # Deterministic algorithm
                edge_mask = torch.ge((similarity + 1.) / 2., self.mp_threshold)
            elif self.inference_mode == 'original':
                edge_mask = torch.ones(similarity.shape).bool()

            else:
                raise NotImplementedError

            x = self.GINE[_](x, edge_index[:, edge_mask], edge_attr[edge_mask])
            if self.inference_mode == 'inter_intra':
                x_intra = self.GINE_intra[_](x, edge_index[:, ~edge_mask], edge_attr[~edge_mask])
                x = x + x_intra

        return x



class RW_Transformer_layer(nn.Module):

    def __init__(self,
                 dim_h, num_heads, dropout,
                 complex_type: str = 'original',
                 similarity_type: str = 'cos',
                 complex_pool_type: str = 'add',
                 force_undirected: bool = True,
                 cluster_threshold: float = 0.1,
                 complex_max_distance: int = 5):
        super().__init__()

        self.similarity_type = similarity_type
        assert complex_type in ['original', 'fixed', 'max_cluster', 'drop_cluster']
        self.complex_type = complex_type
        self.force_undirected = force_undirected
        self.cluster_threshold = cluster_threshold
        self.complex_max_distance = complex_max_distance
        assert complex_pool_type in ['add', 'mean', 'max']
        self.complex_pool_type = complex_pool_type

        self.Transformer = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, edge_index, batch, real_node_mask):
        x_dense = x
        null_node_mask = ~real_node_mask

        if self.complex_type != 'original':
            adj = to_dense_adj(edge_index, batch)
            if self.similarity_type == 'cos':
                similarity = torch.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)
            else:
                raise NotImplementedError
            sim = similarity * (adj.bool())

            if self.complex_type == 'max_cluster':
                mask = (sim == sim.max(dim=-1, keepdim=True)[0]).float()
            elif self.complex_type == 'drop_cluster':
                sim[sim <= self.cluster_threshold] = 0
                mask = sim.bool().float()
            else:
                raise NotImplementedError

            if self.force_undirected:
                mask = (mask + mask.permute(0, 2, 1)).bool().float()
            for i in range(self.complex_max_distance):
                mask = (mask + torch.matmul(mask, mask)).bool().float()

            if self.complex_pool_type == 'add':
                c = torch.einsum('bid,bij->bid', x, mask)
            elif self.complex_pool_type == 'mean':
                mask_sum = mask.sum(dim=-1)
                mask = torch.einsum('bij,bi->bij', mask, 1. / mask_sum)
                c = torch.einsum('bid,bij->bid', x, mask)
            else:  # 'max' & 'attention'
                raise NotImplementedError

            x = x_dense + c

        x = self.Transformer(x, x, x,
                           attn_mask=None,
                           key_padding_mask=null_node_mask,
                           need_weights=False)[0]

        return x[real_node_mask]


# class HRW_MPNN_Transformer_layer(nn.Module):
#
#     def __init__(self,
#                  in_dim: int,
#                  hid_dim: int,
#                  out_dim: int,
#                  num_layer_MPNN: int = 3,
#                  num_layer_Transformer: int = 0,
#                  new_edge_embed: bool = False,
#                  update_edge_embed: bool = False,
#                  max_edgez: int = None,
#                  complex_pool: str = 'add',
#                  drop_ratio: float = 0.0,
#                  attn_drop: float = 0.0,
#                  norm_type: str = 'layer',
#                  JK: str = 'none',
#                  num_head: int = 8,
#                  central_encoding: bool = False,
#                  attn_bias: bool = False,
#                  inference_mode: str = 'sample',  # sample in each layer level or model level?
#                  mp_threshold: float = 0.0,  # used only in averaging method, not needed if use sampling
#                  similarity_type: str = 'cos',
#                  cluster_threshold: float = 1.0):
#         super().__init__()
#
#         self.in_dim = in_dim
#         self.hid_dim = hid_dim
#         self.out_dim = out_dim
#         self.drop_ratio = drop_ratio
#         self.attn_drop = attn_drop
#         self.num_layer_MPNN = num_layer_MPNN
#         self.num_layer_Transformer = num_layer_Transformer
#         self.central_encoding = central_encoding
#         self.attn_bias = attn_bias
#         assert norm_type in ['layer', 'batch']
#         self.norm_type = norm_type
#         self.relu = nn.ReLU(inplace=True)
#         self.graph_mlps = nn.ModuleList([
#             nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True),
#                           nn.Linear(hid_dim, hid_dim)) for _ in range(self.num_layer_MPNN)
#         ])
#         self.graph_norm = nn.ModuleList([nn.LayerNorm(hid_dim) if self.norm_type == 'layer'
#                                          else nn.BatchNorm1d(hid_dim) for _ in range(self.num_layer_MPNN)])
#         self.graph_ffn = nn.ModuleList([
#             nn.Sequential(nn.Linear(hid_dim, 2*hid_dim), nn.ReLU(inplace=True),
#                           nn.Linear(2*hid_dim, hid_dim)) for _ in range(num_layer_Transformer)
#         ])
#         self.graph_ffn_norm = nn.ModuleList([nn.LayerNorm(hid_dim) if self.norm_type == 'layer'
#                                          else nn.BatchNorm1d(hid_dim) for _ in range(self.num_layer_Transformer)])
#         self.graph_self_attn = nn.ModuleList([MultiheadAttention(
#             self.hid_dim, num_head, dropout=attn_drop, batch_first=True) for _ in range(num_layer_Transformer)])
#         self.graph_attn_bias = nn.ModuleList([AttentionBias(num_head) for _ in range(num_layer_Transformer)])
#         self.graph_central_encode = nn.ModuleList([nn.Embedding(10, hid_dim) for _ in range(num_layer_Transformer)])
#         self.graph_attn_norm = nn.ModuleList([nn.LayerNorm(hid_dim) if self.norm_type == 'layer'
#                                          else nn.BatchNorm1d(hid_dim) for _ in range(self.num_layer_MPNN)])
#         self.edge_mlp = nn.Sequential(nn.Linear(hid_dim, hid_dim, bias=False), nn.ReLU(inplace=True),
#                                       nn.Linear(hid_dim, hid_dim, bias=False))
#         self.edge_norm = nn.LayerNorm(hid_dim) if self.norm_type == 'layer' else nn.BatchNorm1d(hid_dim)
#         assert complex_pool in ['max', 'add', 'mean', 'attention', 'mix']
#         self.complex_pool_type = complex_pool
#         self.complex_pool = eval("global_" + complex_pool + "_pool")
#         assert similarity_type in ['cos', 'product', 'project_product', 'project_edge']
#         self.similarity_type = similarity_type
#         assert inference_mode in ['average', 'sample', 'original']
#         self.inference_mode = inference_mode
#         self.mp_threshold = mp_threshold
#         self.cluster_threshold = cluster_threshold
#         self.in_mlp = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(inplace=True))
#         self.out_mlp = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.ReLU(inplace=True))
#         self.update_edge_embed = update_edge_embed
#         self.new_edge_embed = new_edge_embed
#         if max_edgez is not None:
#             self.edge_emb = nn.Embedding(max_edgez + 1, hid_dim, padding_idx=0)
#
#     def num2batch(self, num_subg: Tensor):
#         offset = cumsum_pad0(num_subg)
#         # print(offset.shape, num_subg.shape, offset[-1] + num_subg[-1])
#         batch = torch.zeros((offset[-1] + num_subg[-1]),
#                             device=offset.device,
#                             dtype=offset.dtype)
#         batch[offset] = 1
#         batch[0] = 0
#         batch = batch.cumsum_(dim=0)
#         return batch
#
#     def forward(self, x: Tensor, adj_original: SparseTensor, adj: Tensor, null_node_mask: Tensor, degree: Tensor, distance: Tensor):
#         '''
#         Nm = max(Ni)
#         x: (B, Nm, d)
#         adj: (B, Nm, Nm)
#         subgs: sparse(ns1+ns2+...+nsB, k)
#         num_subg: (B) vector of ns
#         num_node: (B) vector of N
#         '''
#         '''
#         x (ns1+ns2+...+nsB, N_m, (k-1)!, d)
#         '''
#         B, Nm = x.shape[0], x.shape[1]
#         if self.new_edge_embed:
#             if adj_original.dtype == torch.long:
#                 adj = self.edge_emb(adj_original)
#             else:
#                 adj = adj_original.unsqueeze_(-1)  # subadj (B, Nm, Nm, d/1)
#
#         if self.in_dim != self.hid_dim:
#             x = self.in_mlp(x)
#
#         # Done: probabilistic message passing
#         for _ in range(self.num_layer_MPNN):
#             # Random Walk MPNN
#             if self.similarity_type == 'cos':
#                 similarity = torch.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)
#             elif self.similarity_type == 'project_edge':
#                 pro1 = torch.einsum("bijd,bjd->bijd", adj, x)
#                 pro2 = torch.einsum("bijd,bid->bijd", adj, x)
#                 similarity = torch.cosine_similarity(pro1, pro2, dim=-1)
#             else:
#                 raise NotImplementedError
#
#             if self.inference_mode == 'sample':  # MCMC, stochastic sampling method
#                 # TODO: strategy gradient; layer level (equivalent to GAT, probability as weight and mean effect)
#                 #  or model level sampling? replaced by variational inference
#                 sampled = torch.le(torch.rand(B, Nm, Nm).to(x.device), (similarity + 1.) / 2.).float()  # 之后考虑策略梯度
#                 rw_adj = torch.einsum("bijd,bij->bijd", adj, sampled)
#             elif self.inference_mode == 'average':  # Deterministic algorithm
#                 sampled = torch.ge((similarity + 1.) / 2., self.mp_threshold).float()
#                 rw_adj = torch.einsum("bijd,bij->bijd", adj, sampled)
#             elif self.inference_mode == 'original':
#                 rw_adj = adj
#             else:
#                 raise NotImplementedError
#             # mlp(x + relu(einsum))
#             x1 = F.dropout(self.graph_mlps[_](x + torch.einsum("bijd,bjd->bid", rw_adj, x)), self.drop_ratio, self.training)  # (B, Nm, d)
#             x = x + x1
#             x[null_node_mask] = 0
#             if self.norm_type == 'layer':
#                 x = self.graph_norm[_](x)
#             else:
#                 x = self.graph_norm[_](x.reshape(-1, self.hid_dim)).reshape(B, Nm, self.hid_dim)
#
#         # TODO: clustering and intra-cluster pooling
#         # use complexes as prior
#         # v.s. random walk select subgraph
#         # consider different clustering methods, including spectral perspective
#         # TODO: inter-cluster message passing / attention and unpooling
#
#         for _ in range(self.num_layer_Transformer):
#             attn_bias = self.graph_attn_bias[_](distance, adj_original) if self.attn_bias else None
#             x2 = x + self.graph_central_encode[_](degree) if self.central_encoding else x
#             x2 = self.graph_self_attn[_](x2, x2, x2, attn_bias,
#                                          attn_mask=None,
#                                          key_padding_mask=null_node_mask,
#                                          need_weights=False)[0]
#             x2[null_node_mask] = 0
#             if self.norm_type == 'layer':
#                 x2 = self.graph_attn_norm[_](x2)
#             else:
#                 x2 = self.graph_attn_norm[_](x2.reshape(-1, self.hid_dim)).reshape(B, Nm, self.hid_dim)
#             x2 = F.dropout(x2, self.drop_ratio, self.training)
#             x = x + x2
#             x = x + F.dropout(self.graph_ffn[_](x), self.drop_ratio, self.training)
#             x[null_node_mask] = 0
#             if self.norm_type == 'layer':
#                 x = self.graph_ffn_norm[_](x)
#             else:
#                 x = self.graph_ffn_norm[_](x.reshape(-1, self.hid_dim)).reshape(B, Nm, self.hid_dim)
#
#         # TODO: update edge feature
#         if self.update_edge_embed:
#             #     pass
#             if self.similarity_type == 'cos':
#                 similarity = torch.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)
#             elif self.similarity_type == 'project_edge':
#                 pro1 = torch.einsum("bijd,bjd->bijd", adj, x)
#                 pro2 = torch.einsum("bijd,bid->bijd", adj, x)
#                 similarity = torch.cosine_similarity(pro1, pro2, dim=-1)
#             else:
#                 raise NotImplementedError
#             sim = similarity * (adj_original.bool())
#             sampled = torch.lt(torch.rand(B, Nm, Nm).to(x.device), (sim + 1.) / 2.).float()
#             adj1 = x.unsqueeze(-3).repeat(1, Nm, 1, 1)
#             adj2 = x.unsqueeze(-2).repeat(1, 1, Nm, 1)
#             update_adj = torch.einsum("bijd,bij->bijd", adj1 + adj2, sampled)
#             adj = adj + F.dropout(self.edge_mlp(adj + update_adj), self.drop_ratio, self.training)
#             # if self.norm_type == 'layer':
#             #     adj = self.edge_norm(adj)
#             # else:
#             #     adj = self.edge_norm(adj.reshape(-1, self.hid_dim)).reshape(B, Nm, Nm, self.hid_dim)
#             # adj = torch.einsum("bijd,bij->bijd", adj, adj_original.bool().float())
#
#         if self.out_dim != self.hid_dim:
#             x = self.out_mlp(x)
#         x[null_node_mask] = 0
#         return x, adj
