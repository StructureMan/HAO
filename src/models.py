import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from layers import hyp_layers
from layers.hyp_layers import HNNLayer, HyperbolicGraphConvolution, HypLinear
from main import device
import mainfolds

class HAO(nn.Module):
    def __init__(self, feats, n_windows, space):
        super(HAO, self).__init__()
        self.name = 'HAO'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.n_window = n_windows

        class argsR():
            def __init__(self, feat_dim):
                self.manifold = 2
                self.num_layers = 2
                self.act = 'relu'
                self.feat_dim = feat_dim
                self.dim = feat_dim
                self.n_classes = feat_dim
                self.cuda = 1
                self.device = 'cuda:' + str(self.cuda) if int(self.cuda) >= 0 else 'cpu'
                self.manifold = "Euclidean"
                self.model = "HGCN"
                self.c = 1.0  #
                self.task = "rec"
                self.dropout = 0.0
                self.bias = 1
                self.use_att = 1
                self.local_agg = 1
                self.n_heads = 4

        argsR = argsR(self.n_feats)
        self.argsR = argsR
        # 曲率自动改变
        if argsR.c is not None:
            self.c = torch.tensor([argsR.c])
            if not argsR.cuda == -1:
                self.c = self.c.to(argsR.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.0])).to(argsR.device)

        self.manifold = getattr(mainfolds, argsR.manifold)()

        if self.manifold.name == "Hyperboloid":
            argsR.feat_dim = argsR.feat_dim + 1

        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(argsR)

        act = acts[0]
        self.feature_out_layer_dime = self.n_feats

        self.curv_liner = nn.Parameter(torch.Tensor([1.0])).to(argsR.device)

        self.compress_layer = HNNLayer(self.manifold, in_features=argsR.feat_dim,
                                       out_features=self.feature_out_layer_dime,
                                       c=self.curv_liner, act=act, dropout=0.0, use_bias=False).to(device)
        self.compress_dim = 8

        self.curv_t_hgcn_in = nn.Parameter(torch.Tensor([1.0])).to(argsR.device)
        self.curv_t_hgcn_out = nn.Parameter(torch.Tensor([1.0])).to(argsR.device)
        self.t_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, in_features=self.feature_out_layer_dime,
                                                 adj_dim=self.n_window, adj_act=nn.Sigmoid(),
                                                 out_features=self.compress_dim, c_in=self.curv_t_hgcn_in,
                                                 c_out=self.curv_t_hgcn_out, dropout=0.0,
                                                 act=act, use_bias=False, use_att=0, local_agg=0).to(device)

        self.curv_s_hgcn_in = nn.Parameter(torch.Tensor([1.0])).to(argsR.device)
        self.curv_s_hgcn_out = nn.Parameter(torch.Tensor([1.0])).to(argsR.device)

        self.s_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, in_features=self.n_window,
                                                 adj_dim=self.feature_out_layer_dime, adj_act=nn.Sigmoid(),
                                                 out_features=self.compress_dim, c_in=self.curv_s_hgcn_in,
                                                 c_out=self.curv_s_hgcn_out, dropout=0.0,
                                                 act=act, use_bias=False, use_att=0, local_agg=0).to(device)

        self.curv_out = nn.Parameter(torch.Tensor([1.0])).to(argsR.device)
        self.out_layer = HNNLayer(self.manifold,
                                  in_features=self.compress_dim * self.feature_out_layer_dime + self.compress_dim * self.n_window + self.feature_out_layer_dime ** 2 + self.n_window ** 2,
                                  out_features=self.compress_dim * self.feature_out_layer_dime + self.compress_dim * self.n_window + self.feature_out_layer_dime ** 2 + self.n_window ** 2,
                                  c=self.curv_out, act=act, dropout=0.0, use_bias=False).to(device)

        self.fcn_all = nn.Sequential(nn.Linear(
            self.compress_dim * self.feature_out_layer_dime + self.compress_dim * self.n_window + self.feature_out_layer_dime ** 2 + self.n_window ** 2,
            self.n_window * self.n_feats),
            nn.ReLU(True)).to(device)

        self.t_adj = nn.Parameter(torch.rand(self.n_window, self.n_window, dtype=torch.float64, requires_grad=True)).to(
            device)
        self.s_adj = nn.Parameter(
            torch.zeros(self.feature_out_layer_dime, self.feature_out_layer_dime, dtype=torch.float64,
                        requires_grad=True)).to(device)
        self.t_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device) + torch.eye(
            self.n_window).to(device)
        self.s_mask = torch.triu(torch.ones(self.feature_out_layer_dime, self.feature_out_layer_dime), diagonal=1).to(
            device) + torch.eye(self.feature_out_layer_dime).to(device)
        self.t_l = self.n_window ** 2 // 2
        self.s_l = self.n_feats ** 2 // 2

        self.t_adj_w = HypLinear(manifold=self.manifold, in_features=self.n_window, out_features=self.n_window,
                                 c=self.c, dropout=0.0, use_bias=0).to(device)
        self.s_adj_w = HypLinear(manifold=self.manifold, in_features=self.feature_out_layer_dime,
                                 out_features=self.feature_out_layer_dime, c=self.c, dropout=0.0, use_bias=0).to(device)

        self.ls = None

        self.low_curv = [nn.Parameter(torch.Tensor([0.0000001])).to(argsR.device) for _ in range(7)]
        lambda_e = 0.1
        for i in range(self.n_window):
            for j in range(self.n_window):
                if i < j:
                    self.t_adj[i][j] = np.exp(-lambda_e * (j - i))
                if i == j:
                    self.t_adj[i][j] = 1.0
        for i in range(self.feature_out_layer_dime):
            for j in range(self.feature_out_layer_dime):
                if i < j:
                    self.s_adj[i][j] = 1.0
                if i == j:
                    self.s_adj[i][j] = 1.0

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.input_dim, self.output_dim, self.bias, self.c
        )

    def encoder_Feature(self, x, adj):
        x = x.view(-1, self.n_feats)
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder_model.encode(x, adj)
        h = self.decoder_model.decode(h, adj)
        return h

    def generate_adj(self, x):
        x = x.t()
        s_adj = torch.zeros((self.n_feats, self.n_feats), dtype=torch.float64).to(device)
        for i in range(self.n_feats):
            for j in range(i + 1, self.n_feats):
                s_adj[i][j] = self.manifold.sqdist(x[i], x[j], self.c)
        s_adj = s_adj + s_adj.t()
        return x, s_adj

    def update_adj(self):
        with torch.no_grad():
            self.t_adj *= self.t_mask
            self.s_adj *= self.s_mask

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        x = torch.nan_to_num(x, nan=0.)
        if s_adj_hyp is None:
            t_adj_hyp = nn.Parameter(torch.ones(self.n_window, self.n_window), requires_grad=True).to(device).to(
                torch.float64)
            s_adj_hyp = nn.Parameter(torch.ones(self.feature_out_layer_dime, self.feature_out_layer_dime),
                                     requires_grad=True).to(device).to(torch.float64)

        x = x.view(-1, self.n_feats)
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        t_adj_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(t_adj_hyp, self.c), c=self.c),
                                       c=self.c)
        s_adj_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(s_adj_hyp, self.c), c=self.c),
                                       c=self.c)
        x = self.compress_layer(x_hyp)
        t_f, t_adj_hyp = self.t_hgcn((x, t_adj_hyp))
        s_f, s_adj_hyp = self.s_hgcn((x.t(), s_adj_hyp))
        x = torch.cat((t_f.view(-1), s_f.view(-1), t_adj_hyp.view(-1), s_adj_hyp.view(-1))).view(1, -1)
        x = self.out_layer(x)
        out = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c).to(device)
        t_adj_hyp = F.sigmoid(out.view(-1)[-(self.n_window ** 2 + self.feature_out_layer_dime ** 2): -(
                self.n_window ** 2 + self.feature_out_layer_dime ** 2) + self.n_window ** 2]).detach()
        s_adj_hyp = F.sigmoid(out.view(-1)[-(self.feature_out_layer_dime ** 2):]).detach()
        x = self.fcn_all(out)
        return x.view(-1), t_adj_hyp.view(self.n_window, -1), s_adj_hyp.view(self.feature_out_layer_dime,
                                                                             -1), torch.stack(
            [self.c, self.curv_liner, self.curv_t_hgcn_in, self.curv_s_hgcn_in, self.curv_t_hgcn_out,
             self.curv_s_hgcn_out, self.curv_out])
