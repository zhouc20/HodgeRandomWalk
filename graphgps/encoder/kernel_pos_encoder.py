import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder, register_edge_encoder


class KernelPENodeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding node encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = getattr(cfg, f"posenc_{self.kernel_type}")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        num_rw_steps = len(pecfg.kernel.times)
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 1:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if self.kernel_type == 'InterRWSE':
            pestat_var = f"pestat_RWSE"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch


@register_node_encoder('RWSE')
class RWSENodeEncoder(KernelPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'RWSE'


@register_node_encoder('InterRWSE_Node')
class InterRWSENodeEncoder(KernelPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'InterRWSE'


@register_node_encoder('HKdiagSE')
class HKdiagSENodeEncoder(KernelPENodeEncoder):
    """Heat kernel (diagonal) Structural Encoding node encoder.
    """
    kernel_type = 'HKdiagSE'


@register_node_encoder('ElstaticSE')
class ElstaticSENodeEncoder(KernelPENodeEncoder):
    """Electrostatic interactions Structural Encoding node encoder.
    """
    kernel_type = 'ElstaticSE'


class KernelPEEdgeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding edge encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = getattr(cfg, f"posenc_{self.kernel_type}")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        bias = pecfg.bias
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable
        if self.kernel_type == 'HodgeLap1PE':
            self.embed_type = pecfg.embed_type  #
            assert self.embed_type in ['sum_zero_abs', 'proj_zero', 'proj_low']
            if self.embed_type == 'sum_zero_abs':
                indim = 1  # sum over all eigenvectors
            elif self.embed_type == 'proj_zero':
                indim = dim_emb - dim_pe
            elif self.embed_type == 'proj_low':
                indim = 1
                dim_zero = pecfg.dim_zero
                dim_low = pecfg.dim_low
                dim_all = pecfg.dim_all
                assert dim_all + dim_low + dim_zero == dim_pe
            else:
                raise NotImplementedError
        else:
            indim = len(pecfg.kernel.times)

        if dim_emb - dim_pe < 1:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(indim)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(indim, dim_pe, bias=bias))
                layers.append(activation())
            else:
                layers.append(nn.Linear(indim, 2 * dim_pe, bias=bias))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe, bias=bias))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe, bias=bias))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(indim, dim_pe, bias=bias)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

        if self.kernel_type == 'HodgeLap1PE':
            if self.embed_type == 'proj_low':
                self.max_zero_freq = pecfg.max_zero_freq
                self.max_low_freq = pecfg.max_low_freq
                self.max_total_freq = pecfg.max_total_freq
                if model_type == 'mlp':
                    layers = []
                    if n_layers == 1:
                        layers.append(nn.Linear(2, dim_all, bias=bias))
                        layers.append(activation())
                    else:
                        layers.append(nn.Linear(2, 2 * dim_pe, bias=bias))
                        layers.append(activation())
                        for _ in range(n_layers - 2):
                            layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe, bias=bias))
                            layers.append(activation())
                        layers.append(nn.Linear(2 * dim_pe, dim_all, bias=bias))
                        layers.append(activation())
                    self.all_encoder = nn.Sequential(*layers)
                elif model_type == 'linear':
                    self.all_encoder = nn.Linear(2, dim_all, bias=bias)
                else:
                    raise ValueError(f"{self.__class__.__name__}: Does not support "
                                     f"'{model_type}' encoder model.")

                if model_type == 'mlp':
                    layers = []
                    if n_layers == 1:
                        layers.append(nn.Linear(1, dim_low, bias=bias))
                        layers.append(activation())
                    else:
                        layers.append(nn.Linear(1, 2 * dim_pe, bias=bias))
                        layers.append(activation())
                        for _ in range(n_layers - 2):
                            layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe, bias=bias))
                            layers.append(activation())
                        layers.append(nn.Linear(2 * dim_pe, dim_low, bias=bias))
                        layers.append(activation())
                    self.low_encoder = nn.Sequential(*layers)
                elif model_type == 'linear':
                    self.low_encoder = nn.Linear(1, dim_low, bias=bias)
                else:
                    raise ValueError(f"{self.__class__.__name__}: Does not support "
                                     f"'{model_type}' encoder model.")

                if model_type == 'mlp':
                    layers = []
                    if n_layers == 1:
                        layers.append(nn.Linear(1, dim_zero, bias=bias))
                        layers.append(activation())
                    else:
                        layers.append(nn.Linear(1, 2 * dim_pe, bias=bias))
                        layers.append(activation())
                        for _ in range(n_layers - 2):
                            layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe, bias=bias))
                            layers.append(activation())
                        layers.append(nn.Linear(2 * dim_pe, dim_zero, bias=bias))
                        layers.append(activation())
                    self.zero_encoder = nn.Sequential(*layers)
                elif model_type == 'linear':
                    self.zero_encoder = nn.Linear(1, dim_zero, bias=bias)
                else:
                    raise ValueError(f"{self.__class__.__name__}: Does not support "
                                     f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if self.kernel_type == 'InterRWSE':
            pestat_var = f"pestat_EdgeRWSE"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num edges) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num edges) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        if self.kernel_type == 'HodgeLap1PE':
            if self.embed_type == 'sum_zero_abs':
                pos_enc = self.pe_encoder(pos_enc)
            elif self.embed_type == 'proj_zero':
                proj = torch.matmul(pos_enc, pos_enc.permute(1, 0))
                mask = torch.zeros([pos_enc.shape[0], pos_enc.shape[0]], dtype=torch.float, device=pos_enc.device)
                acc = 0
                for i in range(batch.num_graphs):
                    mask[acc: acc+batch.num_undir_edges[i], acc: acc+batch.num_undir_edges[i]] = 1
                    acc += batch.num_undir_edges[i]
                proj = proj * mask
                pos_enc = torch.matmul(proj, batch.edge_attr)
                pos_enc = self.pe_encoder(pos_enc)  # (Num edges) x dim_pe

            elif self.embed_type == 'proj_low':
                x_zero = self.zero_encoder(pos_enc[:, 0].unsqueeze(1))
                x_low = self.low_encoder(pos_enc[:, 1].unsqueeze(1))
                x_all = self.all_encoder(pos_enc[:, 2:].reshape(pos_enc.shape[0], self.max_total_freq, 2))
                x_all = torch.sum(x_all, dim=1, keepdim=False)
                pos_enc = torch.cat([x_zero, x_low, x_all], dim=1)
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.edge_attr)
        else:
            h = batch.edge_attr
        # Concatenate final PEs to input embedding
        batch.edge_attr = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch


@register_edge_encoder('HodgeLap1PE')
class HodgeLap1PEEdgeEncoder(KernelPEEdgeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'HodgeLap1PE'


@register_edge_encoder('EdgeRWSE')
class EdgeRWSEEdgeEncoder(KernelPEEdgeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'EdgeRWSE'


@register_edge_encoder('InterRWSE_Edge')
class InterRWSEEdgeEncoder(KernelPEEdgeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'InterRWSE'
