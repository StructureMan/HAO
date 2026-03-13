import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
import torch_geometric.nn
from dgl.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops
from torch.nn import TransformerEncoder, init
from torch.nn import TransformerDecoder
from layers import hyp_layers
from layers.hyp_layers import HNNLayer, HyperbolicGraphConvolution, HypLinear
from layers.layers import FermiDiracDecoder
from main import device
from src.dlutils import *
from src.constants import *
import mainfolds
import models_h.encoders as encoders
from models_h.decoders import model2decoder
torch.autograd.set_detect_anomaly(True)



## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2).to(device)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        ).to(device)

    def forward(self, x, hidden=None):
        hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64).to(device) if hidden is not None else hidden
        out, hidden = self.lstm(x.to(torch.float64).view(1, 1, -1), hidden)
        ## Encode
        x = self.encoder(out.to(torch.float64))
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5 * logvar).to(torch.float64)
        eps = torch.randn_like(std).to(torch.float64)
        x = mu + eps * std
        ## Decoder
        x = self.decoder(x)
        return x.view(-1), mu.view(-1), logvar.view(-1), hidden.to(device)



class ParallelBranchNetwork(nn.Module):
    def __init__(self, nfeats, n_window, act=nn.ReLU(), space_name="", is_change_branch=True):
        """
        Initialize the ParallelBranchNetwork module with multiple parallel branches for feature extraction.

        Args:
            nfeats: Number of input features.
            n_window: Window size for certain branches.
            act: Activation function to use (default is ReLU).
            space_name: Name of the space (e.g., Euclidean, Hyperboloid).
            is_change_branch: Boolean flag to determine whether to include the 'smallest_branch' (default is True).
        """
        super().__init__()
        self.gain = math.sqrt(2)  # Gain factor for weight initialization
        self.space_name = space_name
        self.is_change_branch = is_change_branch

        # Define parallel branches for feature extraction
        if self.is_change_branch:
            # Periodic branch: captures periodic patterns
            self.periodic_branch = nn.Sequential(
                nn.Linear(2 * nfeats, n_window * nfeats), act,
                nn.Linear(n_window * nfeats, nfeats), act,
            )
            # Mid branch: captures intermediate-level patterns
            self.mid_branch = nn.Sequential(
                nn.Linear(2 * nfeats, n_window + nfeats), act,
                nn.Linear(n_window + nfeats, nfeats), act,
            )
            # Abrupt branch: captures sudden changes
            self.abrupt_branch = nn.Sequential(
                nn.Linear(2 * nfeats, n_window), act,
                nn.Linear(n_window, nfeats), act,
            )
            # Mini branch: captures fine-grained details
            self.mini_branch = nn.Sequential(
                nn.Linear(2 * nfeats, nfeats), act,
                nn.Linear(nfeats, nfeats), act,
            )
            # Smallest branch: captures very fine details (only included if is_change_branch is True)
            self.smallest_branch = nn.Sequential(
                nn.Linear(2 * nfeats, 64), act,
                nn.Linear(64, nfeats), act,
            )
            # Constant branch: captures baseline patterns
            self.constrant_brach = nn.Sequential(
                nn.Linear(2 * nfeats, 32), act,
                nn.Linear(32, nfeats), act,
            )
            # Fusion weights for combining branch outputs
            self.fusion_weights = nn.Parameter(torch.ones(6))
        else:
            # Similar structure but without the 'smallest_branch'
            self.periodic_branch = nn.Sequential(
                nn.Linear(2 * nfeats, n_window * nfeats), act,
                nn.Linear(n_window * nfeats, nfeats), act,
            )
            self.mid_branch = nn.Sequential(
                nn.Linear(2 * nfeats, n_window + nfeats), act,
                nn.Linear(n_window + nfeats, nfeats), act,
            )
            self.abrupt_branch = nn.Sequential(
                nn.Linear(2 * nfeats, n_window), act,
                nn.Linear(n_window, nfeats), act,
            )
            self.mini_branch = nn.Sequential(
                nn.Linear(2 * nfeats, nfeats), act,
                nn.Linear(nfeats, nfeats), act,
            )
            self.constrant_brach = nn.Sequential(
                nn.Linear(2 * nfeats, 32), act,
                nn.Linear(32, nfeats), act,
            )
            # Fusion weights for combining branch outputs (5 branches instead of 6)
            self.fusion_weights = nn.Parameter(torch.ones(5))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all linear layers using Kaiming uniform initialization.
        """
        # Initialize weights for each branch
        for branch_name in ['periodic_branch', 'mid_branch', 'smallest_branch', 'constrant_brach', 'mini_branch', 'abrupt_branch']:
            if hasattr(self, branch_name):
                for layer in getattr(self, branch_name):
                    if isinstance(layer, nn.Linear):
                        init.kaiming_uniform_(layer.weight, a=self.gain)
                        if layer.bias is not None:
                            fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                            init.uniform_(layer.bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass through the parallel branches and fuse their outputs.

        Args:
            x: Input tensor of shape (batch_size, 2 * nfeats).

        Returns:
            output: Fused output tensor of shape (batch_size, nfeats).
        """
        # Process input through each branch
        if self.is_change_branch:
            periodic_out = self.periodic_branch(x)
            abrupt_out = self.abrupt_branch(x)
            mid_out = self.mid_branch(x)
            mini_branch_out = self.mini_branch(x)
            constrant_brach_out = self.constrant_brach(x)
            smallest_branch_out = self.smallest_branch(x)

            # Normalize fusion weights using softmax
            weights = F.softmax(self.fusion_weights, dim=0)

            # Weighted sum of branch outputs
            output = (
                weights[0] * periodic_out +
                weights[1] * abrupt_out +
                weights[2] * mid_out +
                weights[3] * mini_branch_out +
                weights[4] * constrant_brach_out +
                weights[5] * smallest_branch_out
            )
        else:
            periodic_out = self.periodic_branch(x)
            abrupt_out = self.abrupt_branch(x)
            mid_out = self.mid_branch(x)
            mini_branch_out = self.mini_branch(x)
            constrant_brach_out = self.constrant_brach(x)

            # Normalize fusion weights using softmax
            weights = F.softmax(self.fusion_weights, dim=0)

            # Weighted sum of branch outputs (excluding smallest_branch)
            output = (
                weights[0] * periodic_out +
                weights[1] * abrupt_out +
                weights[2] * mid_out +
                weights[3] * mini_branch_out +
                weights[4] * constrant_brach_out
            )

        return output


class HAO_E(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_E model for hyperbolic anomaly detection in Euclidean space.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_E, self).__init__()
        self.name = 'HAO_E'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Euclidean"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_E model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
class HAO_P(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_P model for hyperbolic anomaly detection in Poincare Ball space.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_P, self).__init__()
        self.name = 'HAO_P'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "PoincareBall"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

        

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

        

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_P model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
class HAO_H(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_H model for hyperbolic anomaly detection in Hyperboloid space.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_H, self).__init__()
        self.name = 'HAO_H'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Hyperboloid"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_H model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp


# Ablation experiment without HDNN
class HAO_E_HDNN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_E_HDNN model for hyperbolic anomaly detection in Euclidean space,
        excluding the HDNN (Hyperbolic Deep Neural Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_E_HDNN, self).__init__()
        self.name = 'HAO_E_HDNN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Euclidean"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_E_HDNN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        # s_features = self.s_compress_layer(s_features)  # HDNN component removed
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        # t_features = self.t_compress_layer(t_features)  # HDNN component removed
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    


# Ablation experiment without MSCD
class HAO_E_MSCD(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_E_MSCD model for hyperbolic anomaly detection in Euclidean space,
        excluding the MSCD (Multi-Scale Coupling Dynamics) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_E_MSCD, self).__init__()
        self.name = 'HAO_E_MSCD'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Euclidean"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Additional transformation layer (MSCD component removed)
        self.t_layer = nn.Sequential(
            nn.Linear(self.n_feats * 2, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_E_MSCD model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        # out = self.paramodel(x)  # MSCD component removed
        out = self.t_layer(x)  # Direct transformation instead of parallel branches

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp

# Ablation experiment without S-HGCN
class HAO_E_T_HGCN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_E_T_HGCN model for hyperbolic anomaly detection in Euclidean space,
        excluding the S-HGCN (Spatial Hyperbolic Graph Convolution Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_E_T_HGCN, self).__init__()
        self.name = 'HAO_E_T_HGCN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Euclidean"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.n_window,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_E_T_HGCN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine temporal features and residual features
        x = torch.cat([t_gsd_f.t(), RES], dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj_hyp.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without T-HGCN
class HAO_E_S_HGCN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_E_S_HGCN model for hyperbolic anomaly detection in Euclidean space,
        excluding the T-HGCN (Temporal Hyperbolic Graph Convolution Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_E_S_HGCN, self).__init__()
        self.name = 'HAO_E_S_HGCN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Euclidean"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.n_feats,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_E_S_HGCN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Combine spatial features and residual features
        x = torch.cat([s_gsd_f, RES], dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj_hyp.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp


# Ablation experiment without HDNN in Poincare Ball space
class HAO_P_HDNN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_P_HDNN model for hyperbolic anomaly detection in Poincare Ball space,
        excluding the HDNN (Hyperbolic Deep Neural Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_P_HDNN, self).__init__()
        self.name = 'HAO_P_HDNN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "PoincareBall"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_P_HDNN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        # s_features = self.s_compress_layer(s_features)  # HDNN component removed
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        # t_features = self.t_compress_layer(t_features)  # HDNN component removed
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without S-HGCN in Poincare Ball space
class HAO_P_T_HGCN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_P_T_HGCN model for hyperbolic anomaly detection in Poincare Ball space,
        excluding the S-HGCN (Spatial Hyperbolic Graph Convolution Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_P_T_HGCN, self).__init__()
        self.name = 'HAO_P_T_HGCN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "PoincareBall"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.n_window,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_P_T_HGCN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine temporal features and residual features
        x = torch.cat([t_gsd_f.t(), RES], dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj_hyp.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without T-HGCN in Poincare Ball space
class HAO_P_S_HGCN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_P_S_HGCN model for hyperbolic anomaly detection in Poincare Ball space,
        excluding the T-HGCN (Temporal Hyperbolic Graph Convolution Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_P_S_HGCN, self).__init__()
        self.name = 'HAO_P_S_HGCN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "PoincareBall"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.n_feats,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_P_S_HGCN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Combine spatial features and residual features
        x = torch.cat([s_gsd_f, RES], dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj_hyp.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without AHGSD in Poincare Ball space
class HAO_P_AHGSD(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_P_AHGSD model for hyperbolic anomaly detection in Poincare Ball space,
        excluding the AHGSD (Adaptive Hyperbolic Graph Structure Discovery) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_P_AHGSD, self).__init__()
        self.name = 'HAO_P_AHGSD'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "PoincareBall"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_P_AHGSD model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)

        # Temporal feature learning with hyperbolic compression
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.mul(s_features, t_features.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj_hyp.view(self.n_window, -1), t_adj_hyp.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without MSCD in Poincare Ball space
class HAO_P_MSCD(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_P_MSCD model for hyperbolic anomaly detection in Poincare Ball space,
        excluding the MSCD (Multi-Structure Coupling Discovery) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_P_MSCD, self).__init__()
        self.name = 'HAO_P_MSCD'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "PoincareBall"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Additional transformation layer for temporal features
        self.t_layer = nn.Sequential(
            nn.Linear(self.n_feats * 2, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_P_MSCD model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        out = self.t_layer(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp

# Ablation experiment without HDNN in Hyperboloid space
class HAO_H_HDNN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_H_HDNN model for hyperbolic anomaly detection in Hyperboloid space,
        excluding the HDNN (Hyperbolic Deep Neural Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_H_HDNN, self).__init__()
        self.name = 'HAO_H_HDNN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Hyperboloid"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_H_HDNN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        # s_features = self.s_compress_layer(s_features)  # HDNN component removed
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        # t_features = self.t_compress_layer(t_features)  # HDNN component removed
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without S-HGCN in Hyperboloid space
class HAO_H_T_HGCN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_H_T_HGCN model for hyperbolic anomaly detection in Hyperboloid space,
        excluding the S-HGCN (Spatial Hyperbolic Graph Convolution Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_H_T_HGCN, self).__init__()
        self.name = 'HAO_H_T_HGCN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Hyperboloid"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.n_window,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_H_T_HGCN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine temporal features and residual features
        x = torch.cat([t_gsd_f.t(), RES], dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj_hyp.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without T-HGCN in Hyperboloid space
class HAO_H_S_HGCN(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_H_S_HGCN model for hyperbolic anomaly detection in Hyperboloid space,
        excluding the T-HGCN (Temporal Hyperbolic Graph Convolution Network) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_H_S_HGCN, self).__init__()
        self.name = 'HAO_H_S_HGCN'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Hyperboloid"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.n_feats,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_H_S_HGCN model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Combine spatial features and residual features
        x = torch.cat([s_gsd_f, RES], dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj_hyp.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without HGCN in Hyperboloid space
class HAO_H_AHGSD(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_H_AHGSD model for hyperbolic anomaly detection in Hyperboloid space,
        excluding the HGCN (Hyperbolic Graph Convolution Network) components.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_H_AHGSD, self).__init__()
        self.name = 'HAO_H_AHGSD'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Hyperboloid"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_H_AHGSD model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)

        # Temporal feature learning with hyperbolic compression
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.mul(s_features, t_features.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj_hyp.view(self.n_window, -1), t_adj_hyp.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without HGCN in Euclidean space
class HAO_E_AHGSD(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_E_AHGSD model for hyperbolic anomaly detection in Euclidean space,
        excluding the HGCN (Hyperbolic Graph Convolution Network) components.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_E_AHGSD, self).__init__()
        self.name = 'HAO_E_AHGSD'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Euclidean"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_E_AHGSD model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)

        # Temporal feature learning with hyperbolic compression
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.mul(s_features, t_features.t()), RES), dim=1)
        out = self.paramodel(x)

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj_hyp.view(self.n_window, -1), t_adj_hyp.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp
    
# Ablation experiment without MSCD in Hyperboloid space
class HAO_H_MSCD(nn.Module):
    def __init__(self, feats, n_windows, is_change_branch):
        """
        Initialize the HAO_H_MSCD model for hyperbolic anomaly detection in Hyperboloid space,
        excluding the MSCD (Multi-Scale Coupling Dynamics) component.

        Args:
            feats: Number of input features.
            n_windows: Window size for temporal processing.
            is_change_branch: Boolean flag to determine whether to include additional branches.
        """
        super(HAO_H_MSCD, self).__init__()
        self.name = 'HAO_H_MSCD'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_window = n_windows
        self.space_name = "Hyperboloid"
        self.manifold = getattr(mainfolds, self.space_name)()  # Manifold type: Euclidean, Hyperboloid, PoincareBall
        self.act = nn.ReLU()
        self.use_bias = False
        self.dropout = 0.0
        self.gain = math.sqrt(2)
        self.is_change_branch = is_change_branch

        # Curvature initialization for hyperbolic layers
        self.curv_liner_s = nn.Parameter(torch.Tensor([1.0]))  # Curvature for spatial compression
        self.curv_liner_t = nn.Parameter(torch.Tensor([1.0]))  # Curvature for temporal compression
        self.curv_t_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for temporal HGCN
        self.curv_t_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for temporal HGCN
        self.curv_s_HGCN_in = nn.Parameter(torch.Tensor([1.0]))  # Input curvature for spatial HGCN
        self.curv_s_HGCN_out = nn.Parameter(torch.Tensor([1.0]))  # Output curvature for spatial HGCN
        self.compress_dim = 16  # Dimension for compressed features

        # Linear compression layers for spatial and temporal features
        self.s_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_feats, out_features=self.n_feats,
            gain=self.gain, c=self.curv_liner_s, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        self.t_compress_layer = HNNLayer(
            self.manifold, in_features=self.n_window, out_features=self.n_window,
            gain=self.gain, c=self.curv_liner_t, act=self.act, dropout=self.dropout,
            use_bias=self.use_bias
        ).to(device)

        # Hyperbolic Graph Convolution layers for spatial and temporal aggregation
        self.s_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_feats, adj_dim=self.n_feats,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_s_HGCN_in, c_out=self.curv_s_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        self.t_HGCN = HyperbolicGraphConvolution(
            manifold=self.manifold, in_features=self.n_window, adj_dim=self.n_window,
            adj_act=self.act, gain=self.gain, out_features=self.compress_dim,
            c_in=self.curv_t_HGCN_in, c_out=self.curv_t_HGCN_out, dropout=self.dropout,
            act=self.act, use_bias=self.use_bias, use_att=0, local_agg=0
        ).to(device)

        # Mask matrices for adjacency relationships
        self.t_mask = torch.triu(torch.ones(self.n_feats, self.n_feats), diagonal=1).to(device)
        self.s_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device)

        # Parallel branch network for feature fusion
        self.paramodel = ParallelBranchNetwork(
            nfeats=self.n_feats, n_window=self.n_window, space_name=self.space_name,
            is_change_branch=self.is_change_branch
        ).to(device)

        # Residual layer for feature refinement
        self.res_layer = nn.Sequential(
            nn.Linear(self.n_feats, self.n_feats), self.act
        ).to(device)

        # Additional transformation layer (MSCD component removed)
        self.t_layer = nn.Sequential(
            nn.Linear(self.n_feats * 2, self.n_feats), self.act
        ).to(device)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of all layers using Kaiming uniform initialization.
        """
        # Initialize weights for HNNLayer modules
        if hasattr(self, 's_compress_layer'):
            self.s_compress_layer.reset_parameters()
        if hasattr(self, 't_compress_layer'):
            self.t_compress_layer.reset_parameters()

        # Initialize weights for HyperbolicGraphConvolution modules
        if hasattr(self, 's_HGCN'):
            self.s_HGCN.reset_parameters()
        if hasattr(self, 't_HGCN'):
            self.t_HGCN.reset_parameters()

        # Initialize weights for residual layer
        if hasattr(self, 'res_layer'):
            for layer in self.res_layer:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=self.gain)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(layer.bias, -bound, bound)

        # Initialize weights for parallel branch network
        self.paramodel.reset_parameters()

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward pass through the HAO_H_MSCD model.

        Args:
            x: Input tensor of shape (batch_size, n_feats).
            t_adj_hyp: Optional temporal adjacency matrix.
            s_adj_hyp: Optional spatial adjacency matrix.

        Returns:
            out: Output tensor of shape (batch_size, n_feats).
            s_adj: Updated spatial adjacency matrix.
            t_adj: Updated temporal adjacency matrix.
            curvatures: Stack of curvature parameters.
            t_adj_hyp: Temporal adjacency matrix.
            s_adj_hyp: Spatial adjacency matrix.
        """
        # Reshape input and extract spatial/temporal features
        x = x.view(-1, self.n_feats)
        s_features, t_features = x, x.t()
        RES = self.res_layer(x)

        # Initialize adjacency matrices if not provided
        if t_adj_hyp is None:
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Spatial feature learning with hyperbolic compression and aggregation
        s_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(s_features, c=self.curv_liner_s), c=self.curv_liner_s),
            c=self.curv_liner_s
        )
        s_features = self.s_compress_layer(s_features)
        s_gsd_f, _ = self.s_HGCN((s_features, s_adj_hyp))
        if self.training:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        else:
            s_update_adj = self.manifold.sqdistmatrix(s_gsd_f, self.curv_s_HGCN_out, s_adj_hyp)
        s_adj = nn.Parameter(s_update_adj * self.s_mask, requires_grad=True).to(device).to(torch.float64)
        s_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(s_gsd_f, c=self.curv_s_HGCN_out), c=self.curv_s_HGCN_out)

        # Temporal feature learning with hyperbolic compression and aggregation
        t_features = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(t_features, c=self.curv_liner_t), c=self.curv_liner_t),
            c=self.curv_liner_t
        )
        t_features = self.t_compress_layer(t_features)
        t_gsd_f, _ = self.t_HGCN((t_features, t_adj_hyp))
        if self.training:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        else:
            t_update_adj = self.manifold.sqdistmatrix(t_gsd_f, self.curv_t_HGCN_out, t_adj_hyp)
        t_adj = nn.Parameter(t_update_adj * self.t_mask, requires_grad=True).to(device).to(torch.float64)
        t_gsd_f = self.manifold.proj_tan0(self.manifold.logmap0(t_gsd_f, c=self.curv_t_HGCN_out), c=self.curv_t_HGCN_out)

        # Combine spatial-temporal features and residual features
        x = torch.cat((torch.matmul(s_gsd_f, t_gsd_f.t()), RES), dim=1)
        # out = self.paramodel(x)  # MSCD component removed
        out = self.t_layer(x)  # Direct transformation instead of parallel branches

        # Return outputs and updated adjacency matrices
        return out.view(-1), s_adj.view(self.n_window, -1), t_adj.view(self.n_feats, -1), torch.stack(
            [self.curv_liner_s, self.curv_liner_t, self.curv_t_HGCN_in, self.curv_s_HGCN_in, self.curv_t_HGCN_out,
             self.curv_s_HGCN_out]
        ), t_adj_hyp, s_adj_hyp






## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
    def __init__(self, feats):
        super(MSCRED, self).__init__()
        self.name = 'MSCRED'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.ModuleList([
            ConvLSTM(1, 32, (3, 3), 1, True, True, False),
            ConvLSTM(32, 64, (3, 3), 1, True, True, False),
            ConvLSTM(64, 128, (3, 3), 1, True, True, False),
        ]
        ).to(device)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
        ).to(device)

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        for cell in self.encoder:
            _, z = cell(z.view(1, *z.shape))
            z = z[0][0]
        ## Decode
        x = self.decoder(z)
        return x.view(-1)




## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
    def __init__(self, feats):
        super(CAE_M, self).__init__()
        self.name = 'CAE_M'
        self.lr = 0.001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
        ).to(device)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
        ).to(device)

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        z = self.encoder(z)
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
    def __init__(self, feats):
        super(MTAD_GAT, self).__init__()
        self.name = 'MTAD_GAT'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.n_hidden = feats * feats
        self.g = dgl.graph((torch.tensor(list(range(1, feats + 1))), torch.tensor([0] * feats))).to(device)
        self.g = dgl.add_self_loop(self.g).to(device)
        self.feature_gat = GATConv(feats, 1, feats).to(device)
        self.time_gat = GATConv(feats, 1, feats).to(device)
        self.gru = nn.GRU((feats + 1) * feats * 3, feats * feats, 1).to(device)

    def forward(self, data, hidden):
        hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64).to(device) if hidden is not None else hidden
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.cat((torch.zeros(1, self.n_feats).to(device), data))
        feat_r = self.feature_gat(self.g, data_r)
        data_t = torch.cat((torch.zeros(1, self.n_feats).to(device), data.t()))
        time_r = self.time_gat(self.g, data_t)
        data = torch.cat((torch.zeros(1, self.n_feats).to(device), data))
        data = data.view(self.n_window + 1, self.n_feats, 1)
        x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
        x, h = self.gru(x, hidden)
        return x.view(-1), h

## GDN Model (AAAI 21)
class GDN(nn.Module):
    def __init__(self, feats):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats
        src_ids = np.repeat(np.array(list(range(feats))), feats)
        dst_ids = np.array(list(range(feats)) * feats)
        self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids))).to(device)
        self.g = dgl.add_self_loop(self.g).to(device)
        self.feature_gat = GATConv(1, 1, feats).to(device)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
        ).to(device)
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
        ).to(device)

    def forward(self, data):
        # Bahdanau style attention
        att_score = self.attention(data.to(torch.float64)).view(self.n_window, 1)
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.matmul(data.to(torch.float64).permute(1, 0), att_score.to(torch.float64))
        # GAT convolution on complete graph
        feat_r = self.feature_gat(self.g, data_r.to(torch.float64))
        feat_r = feat_r.view(self.n_feats, self.n_feats)
        # Pass through a FCN
        x = self.fcn(feat_r.to(torch.float64))
        return x.view(-1)

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
    def __init__(self, feats):
        super(MAD_GAN, self).__init__()
        self.name = 'MAD_GAN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_window = 5  # MAD_GAN w_size = 5
        self.n = self.n_feats * self.n_window
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        ).to(device)
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        ).to(device)

    def forward(self, g):
        ## Generate
        z = self.generator(g.view(1, -1))
        ## Discriminator
        real_score = self.discriminator(g.view(1, -1))
        fake_score = self.discriminator(z.view(1, -1))
        return z.view(-1), real_score.view(-1), fake_score.view(-1)




# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window).to(device)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1).to(
            device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1).to(device)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1).to(
            device)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1).to(device)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1).to(
            device)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1).to(device)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid()).to(device)

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2
