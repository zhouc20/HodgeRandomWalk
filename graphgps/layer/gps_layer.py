import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.rwgnn_layer import RW_MPNN_layer, RW_Transformer_layer


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False, num_layer_MPNN=1, similarity_type='cos',
                 inference_mode='original', mp_threshold=0.0, force_undirected=False, complex_type='original',
                 complex_pool_type='add', cluster_threshold=0.1, complex_max_distance=5):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.num_layer_MPNN = num_layer_MPNN
        self.similarity_type = similarity_type
        self.inference_mode = inference_mode
        self.force_undirected = force_undirected
        self.mp_threshold = mp_threshold

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type != 'Transformer':
            raise NotImplementedError(
                "Logging of attention weights is only supported for "
                "Transformer global attention model."
            )
        if global_model_type == 'GINE_RW':
            self.inference_mode = 'original'

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
            if self.inference_mode in ['inter_intra', 'original_inter']:
                self.local_model_intra = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
                if self.inference_mode in ['inter_intra', 'original_inter']:
                    gin_nn_intra = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                           self.activation(),
                                           Linear_pyg(dim_h, dim_h))
                    self.local_model_intra = GINEConvESLapPE(gin_nn_intra)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
                if self.inference_mode in ['inter_intra', 'original_inter']:
                    gin_nn_intra = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                                 self.activation(),
                                                 Linear_pyg(dim_h, dim_h))
                    self.local_model_intra = pygnn.GINEConv(gin_nn_intra)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
            if self.inference_mode in ['inter_intra', 'original_inter']:
                self.local_model_intra = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
            if self.inference_mode in ['inter_intra', 'original_inter']:
                self.local_model_intra = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
            if self.inference_mode in ['inter_intra', 'original_inter']:
                self.local_model_intra = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        elif local_gnn_type == 'GINE_RW':
            gin_nn = nn.ModuleList([nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h)) for _ in range(num_layer_MPNN)])
            gin_nn_2 = nn.ModuleList([nn.Sequential(Linear_pyg(dim_h, dim_h),
                                                  self.activation(),
                                                  Linear_pyg(dim_h, dim_h)) for _ in range(num_layer_MPNN)])
            self.local_model = RW_MPNN_layer(gin_nn, gin_nn_2, num_layer_MPNN, similarity_type, inference_mode, mp_threshold, force_undirected)

        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'GINE_RW':
            gin_nn = nn.ModuleList([nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h)) for _ in range(num_layer_MPNN)])
            gin_nn_2 = nn.ModuleList([nn.Sequential(Linear_pyg(dim_h, dim_h),
                                                    self.activation(),
                                                    Linear_pyg(dim_h, dim_h)) for _ in range(num_layer_MPNN)])
            self.self_attn = RW_MPNN_layer(gin_nn, gin_nn_2, num_layer_MPNN, similarity_type, inference_mode, mp_threshold, force_undirected)

        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'RW_Transformer':
            self.self_attn = RW_Transformer_layer(dim_h, num_heads, self.attn_dropout, complex_type,
                                                  similarity_type, complex_pool_type, force_undirected,
                                                  cluster_threshold, complex_max_distance)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        edge_index = batch.edge_index
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                if self.inference_mode == 'original':
                    local_out = self.local_model(Batch(batch=batch,
                                                       x=h,
                                                       edge_index=batch.edge_index,
                                                       edge_attr=batch.edge_attr,
                                                       pe_EquivStableLapPE=es_data))
                    # GatedGCN does residual connection and dropout internally.
                    h_local = local_out.x
                    batch.edge_attr = local_out.edge_attr

                elif self.inference_mode in ['sample', 'inter_intra', 'average', 'original_inter']:  # MCMC, stochastic sampling method
                    # TODO: strategy gradient; layer level (equivalent to GAT, probability as weight and mean effect)
                    #  or model level sampling? replaced by variational inference
                    if self.similarity_type == 'cos':
                        x_src = h[edge_index[0]]
                        x_des = h[edge_index[1]]
                        similarity = torch.cosine_similarity(x_src, x_des, dim=-1)
                    else:
                        raise NotImplementedError
                    rdn = torch.rand(similarity.shape).to(h.device)
                    edge_mask = torch.le(rdn, (similarity + 1.) / 2.)  # 之后考虑策略梯度
                    if self.inference_mode == 'average':  # Deterministic algorithm
                        edge_mask = torch.ge((similarity + 1.) / 2., self.mp_threshold)
                    local_out = self.local_model(Batch(batch=batch,
                                                       x=h,
                                                       edge_index=batch.edge_index[:, edge_mask],
                                                       edge_attr=batch.edge_attr[edge_mask],
                                                       pe_EquivStableLapPE=es_data))
                    # GatedGCN does residual connection and dropout internally.
                    if self.inference_mode in ['sample', 'average']:
                        h_local = local_out.x
                        batch.edge_attr[edge_mask] = local_out.edge_attr
                    elif self.inference_mode == 'original_inter':
                        local_out_intra = self.local_model(Batch(batch=batch,
                                                           x=h,
                                                           edge_index=batch.edge_index,
                                                           edge_attr=batch.edge_attr,
                                                           pe_EquivStableLapPE=es_data))
                        h_local = local_out.x + local_out_intra.x
                        batch.edge_attr = local_out_intra.edge_attr
                        batch.edge_attr[edge_mask] = batch.edge_attr[edge_mask] + local_out.edge_attr
                    else:
                        local_out_intra = self.local_model(Batch(batch=batch,
                                                           x=h,
                                                           edge_index=batch.edge_index[:, ~edge_mask],
                                                           edge_attr=batch.edge_attr[~edge_mask],
                                                           pe_EquivStableLapPE=es_data))
                        h_local = local_out.x + local_out_intra.x
                        batch.edge_attr[edge_mask] = local_out.edge_attr
                        batch.edge_attr[~edge_mask] = local_out_intra.edge_attr
                else:
                    raise NotImplementedError

            elif self.local_gnn_type == 'GINE_RW':
                # max_node = torch.max(torch.diff(batch.ptr))
                h_local = self.local_model(h, batch.edge_index, batch.edge_attr)  # donot need batch
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local

            else:
                if self.inference_mode == 'original':
                    if self.equivstable_pe:
                        h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                                   batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                elif self.inference_mode in ['sample', 'average', 'inter_intra', 'original_inter']:
                    if self.similarity_type == 'cos':
                        x_src = h[edge_index[0]]
                        x_des = h[edge_index[1]]
                        similarity = torch.cosine_similarity(x_src, x_des, dim=-1)
                    else:
                        raise NotImplementedError
                    rdn = torch.rand(similarity.shape).to(h.device)
                    edge_mask = torch.le(rdn, (similarity + 1.) / 2.)  # 之后考虑策略梯度
                    if self.inference_mode == 'average':  # Deterministic algorithm
                        edge_mask = torch.ge((similarity + 1.) / 2., self.mp_threshold)
                    if self.equivstable_pe:
                        h_local = self.local_model(h, batch.edge_index[:, edge_mask], batch.edge_attr[edge_mask],
                                                   batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h, batch.edge_index[:, edge_mask], batch.edge_attr[edge_mask])
                    if self.inference_mode == 'inter_intra':
                        if self.equivstable_pe:
                            h_local_intra = self.local_model(h, batch.edge_index[:, ~edge_mask], batch.edge_attr[~edge_mask],
                                                       batch.pe_EquivStableLapPE)
                        else:
                            h_local_intra = self.local_model(h, batch.edge_index[:, ~edge_mask], batch.edge_attr[~edge_mask])
                        h_local = h_local + h_local_intra
                    elif self.inference_mode == 'original_inter':
                        if self.equivstable_pe:
                            h_local_intra = self.local_model(h, batch.edge_index, batch.edge_attr,
                                                       batch.pe_EquivStableLapPE)
                        else:
                            h_local_intra = self.local_model(h, batch.edge_index, batch.edge_attr)
                        h_local = h_local + h_local_intra

                else:
                    raise NotImplementedError

                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'GINE_RW':
                h_attn = self.self_attn(h, batch.edge_index, batch.edge_attr)
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
