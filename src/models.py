import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from layers import hyp_layers
from layers.hyp_layers import HNNLayer, HyperbolicGraphConvolution, HypLinear
from main import device
import mainfolds

class HAO_E(nn.Module):
    def __init__(self, feats, n_windows):
        """
        Initialize the HAO_E model.

        Sets up the neural network architecture including encoder, hyperbolic graph convolution layers,
        and decoder components used during the forward pass for feature processing and reconstruction.

        Parameters:
            feats (int): Number of input features (feature dimensionality).
            n_windows (int): Number of time windows, which defines the size of the temporal adjacency matrix.
        """

        super(HAO_E, self).__init__()   

        # Basic model identity and training parameters
        self.name = 'HAO_E'  # Model identifier
        self.lr = 0.002  # Learning rate for optimizer
        self.beta = 0.01  # Regularization coefficient

        # Feature and dimensional configuration
        self.n_feats = feats  # Input feature dimension
        self.n_hidden = 32  # Size of hidden layer in encoder/decoder
        self.n_latent = 8  # Latent space representation size
        self.n_window = n_windows  # Number of time windows for temporal graph
        self.n_latent = self.n_feats  # Latent dimension matches input feature count
        self.compress_dim = 8 * 2  # Dimension after compression in hyperbolic layers

        class argsR():
            """
            Internal configuration class to store model hyperparameters and settings.

            Parameters:
                feat_dim (int): Input feature dimension.

            Attributes:
                manifold (str): Type of geometric space ("Euclidean", "Hyperboloid", or "PoincareBall").
                num_layers (int): Number of network layers.
                act (str): Activation function name.
                feat_dim (int): Feature dimension.
                dim (int): Embedding dimension.
                n_classes (int): Number of output classes (same as input features for reconstruction).
                cuda (int): CUDA device index (-1 for CPU).
                device (str): Device type ('cuda:x' or 'cpu').
                model (str): Model type ("HGCN").
                c (float): Curvature parameter.
                task (str): Task type ("rec" for reconstruction).
                dropout (float): Dropout probability.
                bias (bool): Whether to use bias in layers.
                use_att (bool): Whether to use attention mechanism.
                local_agg (bool): Whether to use local aggregation.
                n_heads (int): Number of attention heads.
            """
            def __init__(self, feat_dim):
                self.manifold = 2
                self.num_layers = 2
                self.act = 'relu'
                self.feat_dim = feat_dim
                self.dim = feat_dim  
                self.n_classes = feat_dim  
                self.cuda = 0
                self.device = 'cuda:' + str(self.cuda) if int(self.cuda) >= 0 else 'cpu'
                self.manifold = "Euclidean"  # [Euclidean, Hyperboloid, PoincareBall]
                self.model = "HGCN"  
                self.c = 1.0  
                self.task = "rec"  
                self.dropout = 0.0
                self.bias = 1
                self.use_att = 1
                self.local_agg = 1
                self.n_heads = 4

        argsR = argsR(self.n_feats)
        self.argsR = argsR  # Store configuration object for later reference

        # Curvature setup
        if argsR.c is not None:
            self.c = torch.tensor([argsR.c])
            if not argsR.cuda == -1:
                self.c = self.c.to(argsR.device)  # Move curvature tensor to GPU if available
        else:
            self.c = nn.Parameter(torch.Tensor([1.0]))  # Learnable curvature parameter

        # Manifold selection based on configuration
        self.manifold = getattr(mainfolds, argsR.manifold)()  # Instantiate specified geometric space

        # Retrieve dimensions, activation functions, and curvatures from helper module
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(argsR)

        act = acts[0]  # Use the first activation function from the list
        self.feature_out_layer_dime = self.n_feats  # Output feature dimension matches input

        # Curvature parameters for different parts of the model
        self.c = nn.Parameter(torch.Tensor([1.0]))
        self.curv_liner = nn.Parameter(torch.Tensor([1.0]))
        self.curv_t_hgcn_in = nn.Parameter(torch.Tensor([1.0]))
        self.curv_t_hgcn_out = nn.Parameter(torch.Tensor([1.0]))
        self.curv_s_hgcn_in = nn.Parameter(torch.Tensor([1.0]))
        self.curv_s_hgcn_out = nn.Parameter(torch.Tensor([1.0]))
        self.curv_out = nn.Parameter(torch.Tensor([1.0]))

        # Compression layer in hyperbolic space
        self.compress_layer = HNNLayer(self.manifold, 
                                    in_features=argsR.feat_dim * 2,
                                    out_features=self.feature_out_layer_dime,
                                    c=self.curv_liner, 
                                    act=act, 
                                    dropout=0.0, 
                                    use_bias=False).to(device)

        # Temporal hyperbolic graph convolution layer
        self.t_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, 
                                                in_features=self.feature_out_layer_dime,
                                                adj_dim=self.n_window, 
                                                adj_act=nn.Sigmoid(),
                                                out_features=self.compress_dim, 
                                                c_in=self.curv_t_hgcn_in,
                                                c_out=self.curv_t_hgcn_out, 
                                                dropout=0.0,
                                                act=act, 
                                                use_bias=False, 
                                                use_att=0, 
                                                local_agg=0).to(device)

        # Spatial hyperbolic graph convolution layer
        self.s_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, 
                                                in_features=self.n_window,
                                                adj_dim=self.feature_out_layer_dime, 
                                                adj_act=nn.Sigmoid(),
                                                out_features=self.compress_dim, 
                                                c_in=self.curv_s_hgcn_in,
                                                c_out=self.curv_s_hgcn_out, 
                                                dropout=0.0,
                                                act=act, 
                                                use_bias=False, 
                                                use_att=0, 
                                                local_agg=0).to(device)

        # Output hyperbolic neural network layer
        self.out_layer = HNNLayer(self.manifold,
                                in_features=self.n_feats,
                                out_features=self.n_feats,
                                c=self.curv_out, 
                                act=act, 
                                dropout=0.0, 
                                use_bias=False).to(device)

        # Fully connected network block
        self.fcn_all = nn.Sequential(
            nn.Linear(self.n_feats * 3, 2 * self.n_latent), 
            nn.SiLU(True),
        ).to(device)

        # Adjacency masks for temporal and spatial graphs
        self.t_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device) + torch.eye(
            self.n_window).to(device)
        self.s_mask = torch.triu(torch.ones(self.feature_out_layer_dime, self.feature_out_layer_dime), diagonal=1).to(
            device) + torch.eye(self.feature_out_layer_dime).to(device)

        # Encoder network: maps input into latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), 
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        ).to(device)

        # Decoder network: reconstructs original input from processed features
        self.decoder = nn.Sequential(
            nn.Linear(3 * self.n_latent, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), 
            nn.ReLU(),
        ).to(device)

        print(f"{self.name} Init Success")  # Notify successful initialization

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward propagation function.

        Processes input data through components like encoder, graph convolutional network, and decoder 
        to generate output features and update graph structure parameters in both temporal and spatial dimensions.

        Args:
            x (torch.Tensor): Input feature matrix of shape (batch_size, n_feats).
            t_adj_hyp (torch.Tensor, optional): Temporal adjacency matrix in hyperbolic space. Defaults to None.
            s_adj_hyp (torch.Tensor, optional): Spatial adjacency matrix in hyperbolic space. Defaults to None.

        Returns:
            torch.Tensor: Final output tensor after decoding (flattened).
            torch.Tensor: Updated temporal adjacency matrix.
            torch.Tensor: Updated spatial adjacency matrix.
            torch.Tensor: Stacked curvature parameters used in the model.
            torch.Tensor: Natural temporal adjacency matrix (unmodified).
            torch.Tensor: Natural spatial adjacency matrix (unmodified).
        """

        # Encode input features into a latent representation
        x = self.encoder(x.view(-1, self.n_feats)).to(torch.float64)
        hidden = x  # Save the encoded input as the hidden state for later use

        # Initialize or reuse temporal and spatial adjacency matrices
        if t_adj_hyp is None:
            # If no temporal adjacency matrix is provided, create a new one using the mask
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            # If provided, initialize with existing values while applying the mask
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Map input features into hyperbolic space
        x_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c
        )

        # Compress the mapped features using a hyperbolic neural network layer
        x = self.compress_layer(x_hyp)

        # Temporal Graph Convolution
        t_f, _ = self.t_hgcn((x, t_adj_hyp))  # Apply temporal hyperbolic graph convolution
        new_T_adj_hyp = self.manifold.sqdistmatrix(t_f, self.c)  # Compute updated temporal adjacency matrix
        new_T_adj_hyp = nn.Parameter(new_T_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)

        # Prepare input for spatial convolution by transposing features
        y_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x.t(), self.c), c=self.c), c=self.c
        )

        # Spatial Graph Convolution
        s_f, _ = self.s_hgcn((y_hyp, s_adj_hyp))  # Apply spatial hyperbolic graph convolution
        new_S_adj_hyp = self.manifold.sqdistmatrix(s_f, self.c)  # Compute updated spatial adjacency matrix
        new_S_adj_hyp = nn.Parameter(new_S_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Feature interaction between temporal and spatial domains
        x = torch.mm(t_f, s_f.t())  # Matrix multiplication to combine temporal and spatial features
        out = self.out_layer(x)  # Pass through output hyperbolic layer

        # Project the output from hyperbolic space back to tangent space
        out = self.manifold.proj_tan0(self.manifold.logmap0(out, c=self.c), c=self.c).to(device)

        # Concatenate the natural hidden representation with processed output
        out = torch.cat((hidden, out), dim=1)

        # Decode the final output to reconstruct input space
        out = self.decoder(out.to(torch.float64))

        # Return the decoded output, updated graph structure parameters,
        # curvature parameters, and v graph structure matrices
        return (
            out.view(-1),  # Flattened output tensor
            new_T_adj_hyp.view(self.n_window, -1),  # Reshaped updated temporal adjacency matrix
            new_S_adj_hyp.view(self.feature_out_layer_dime, -1),  # Reshaped updated spatial adjacency matrix
            torch.stack([  # Stack all curvature parameters
                self.c, self.curv_liner, self.curv_t_hgcn_in, self.curv_s_hgcn_in,
                self.curv_t_hgcn_out, self.curv_s_hgcn_out, self.curv_out
            ]),
            t_adj_hyp,  # natural temporal adjacency matrix
            s_adj_hyp   # natural spatial adjacency matrix
        )
    
class HAO_P(nn.Module):
    def __init__(self, feats, n_windows):
        """
        Initialize the HAO_P model.

        Sets up the neural network architecture including encoder, hyperbolic graph convolution layers,
        and decoder components used during the forward pass for feature processing and reconstruction.

        Parameters:
            feats (int): Number of input features (feature dimensionality).
            n_windows (int): Number of time windows, which defines the size of the temporal adjacency matrix.
        """

        super(HAO_P, self).__init__()

        # Basic model identity and training parameters
        self.name = 'HAO_P'  # Model identifier
        self.lr = 0.002  # Learning rate for optimizer
        self.beta = 0.01  # Regularization coefficient

        # Feature and dimensional configuration
        self.n_feats = feats  # Input feature dimension
        self.n_hidden = 32  # Size of hidden layer in encoder/decoder
        self.n_latent = 8  # Latent space representation size
        self.n_window = n_windows  # Number of time windows for temporal graph
        self.n_latent = self.n_feats  # Latent dimension matches input feature count
        self.compress_dim = 8 * 2  # Dimension after compression in hyperbolic layers

        class argsR():
            """
            Internal configuration class to store model hyperparameters and settings.

            Parameters:
                feat_dim (int): Input feature dimension.

            Attributes:
                manifold (str): Type of geometric space ("Euclidean", "Hyperboloid", or "PoincareBall").
                num_layers (int): Number of network layers.
                act (str): Activation function name.
                feat_dim (int): Feature dimension.
                dim (int): Embedding dimension.
                n_classes (int): Number of output classes (same as input features for reconstruction).
                cuda (int): CUDA device index (-1 for CPU).
                device (str): Device type ('cuda:x' or 'cpu').
                model (str): Model type ("HGCN").
                c (float): Curvature parameter.
                task (str): Task type ("rec" for reconstruction).
                dropout (float): Dropout probability.
                bias (bool): Whether to use bias in layers.
                use_att (bool): Whether to use attention mechanism.
                local_agg (bool): Whether to use local aggregation.
                n_heads (int): Number of attention heads.
            """
            def __init__(self, feat_dim):
                self.manifold = 2
                self.num_layers = 2
                self.act = 'relu'
                self.feat_dim = feat_dim
                self.dim = feat_dim  
                self.n_classes = feat_dim  
                self.cuda = 0
                self.device = 'cuda:' + str(self.cuda) if int(self.cuda) >= 0 else 'cpu'
                self.manifold = "PoincareBall"  # [Euclidean, Hyperboloid, PoincareBall]
                self.model = "HGCN"  
                self.c = 1.0  
                self.task = "rec"  
                self.dropout = 0.0
                self.bias = 1
                self.use_att = 1
                self.local_agg = 1
                self.n_heads = 4

        argsR = argsR(self.n_feats)
        self.argsR = argsR  # Store configuration object for later reference

        # Curvature setup
        if argsR.c is not None:
            self.c = torch.tensor([argsR.c])
            if not argsR.cuda == -1:
                self.c = self.c.to(argsR.device)  # Move curvature tensor to GPU if available
        else:
            self.c = nn.Parameter(torch.Tensor([1.0]))  # Learnable curvature parameter

        # Manifold selection based on configuration
        self.manifold = getattr(mainfolds, argsR.manifold)()  # Instantiate specified geometric space (PoincareBall)

        # Retrieve dimensions, activation functions, and curvatures from helper module
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(argsR)

        act = acts[0]  # Use the first activation function from the list
        self.feature_out_layer_dime = self.n_feats  # Output feature dimension matches input

        # Curvature parameters for different parts of the model
        self.c = nn.Parameter(torch.Tensor([1.0]))
        self.curv_liner = nn.Parameter(torch.Tensor([1.0]))
        self.curv_t_hgcn_in = nn.Parameter(torch.Tensor([1.0]))
        self.curv_t_hgcn_out = nn.Parameter(torch.Tensor([1.0]))
        self.curv_s_hgcn_in = nn.Parameter(torch.Tensor([1.0]))
        self.curv_s_hgcn_out = nn.Parameter(torch.Tensor([1.0]))
        self.curv_out = nn.Parameter(torch.Tensor([1.0]))

        # Compression layer in hyperbolic space
        self.compress_layer = HNNLayer(self.manifold, 
                                    in_features=argsR.feat_dim * 2,
                                    out_features=self.feature_out_layer_dime,
                                    c=self.curv_liner, 
                                    act=act, 
                                    dropout=0.0, 
                                    use_bias=False).to(device)

        # Temporal hyperbolic graph convolution layer
        self.t_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, 
                                                in_features=self.feature_out_layer_dime,
                                                adj_dim=self.n_window, 
                                                adj_act=nn.Sigmoid(),
                                                out_features=self.compress_dim, 
                                                c_in=self.curv_t_hgcn_in,
                                                c_out=self.curv_t_hgcn_out, 
                                                dropout=0.0,
                                                act=act, 
                                                use_bias=False, 
                                                use_att=0, 
                                                local_agg=0).to(device)

        # Spatial hyperbolic graph convolution layer
        self.s_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, 
                                                in_features=self.n_window,
                                                adj_dim=self.feature_out_layer_dime, 
                                                adj_act=nn.Sigmoid(),
                                                out_features=self.compress_dim, 
                                                c_in=self.curv_s_hgcn_in,
                                                c_out=self.curv_s_hgcn_out, 
                                                dropout=0.0,
                                                act=act, 
                                                use_bias=False, 
                                                use_att=0, 
                                                local_agg=0).to(device)

        # Output hyperbolic neural network layer
        self.out_layer = HNNLayer(self.manifold,
                                in_features=self.n_feats,
                                out_features=self.n_feats,
                                c=self.curv_out, 
                                act=act, 
                                dropout=0.0, 
                                use_bias=False).to(device)

        # Fully connected network block
        self.fcn_all = nn.Sequential(
            nn.Linear(self.n_feats * 3, 2 * self.n_latent), 
            nn.SiLU(True),
        ).to(device)

        # Adjacency masks for temporal and spatial graphs
        self.t_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device) + torch.eye(
            self.n_window).to(device)
        self.s_mask = torch.triu(torch.ones(self.feature_out_layer_dime, self.feature_out_layer_dime), diagonal=1).to(
            device) + torch.eye(self.feature_out_layer_dime).to(device)

        # Encoder network: maps input into latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), 
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        ).to(device)

        # Decoder network: reconstructs original input from processed features
        self.decoder = nn.Sequential(
            nn.Linear(3 * self.n_latent, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), 
            nn.ReLU(),
        ).to(device)

        print(f"{self.name} Init Success")  # Notify successful initialization

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward propagation function for the model.

        This function defines how the input data flows through the network to produce an output.
        It involves encoding the input, mapping it into hyperbolic space, performing temporal and spatial graph convolutions,
        combining features, and finally decoding them to reconstruct the original feature space.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, n_feats).
            t_adj_hyp (torch.Tensor, optional): Temporal adjacency matrix in hyperbolic space. Defaults to None.
            s_adj_hyp (torch.Tensor, optional): Spatial adjacency matrix in hyperbolic space. Defaults to None.

        Returns:
            torch.Tensor: The final output tensor after decoding (flattened).
            torch.Tensor: Updated temporal adjacency matrix.
            torch.Tensor: Updated spatial adjacency matrix.
            torch.Tensor: Stacked curvature parameters used throughout the model.
            torch.Tensor: Natural  temporal adjacency matrix.
            torch.Tensor: Natural  spatial adjacency matrix.
        """

        # Encode the input features using the encoder network
        x = self.encoder(x.view(-1, self.n_feats)).to(torch.float64)
        hidden = x  # Store the encoded representation for later use

        # Initialize or reuse the temporal and spatial adjacency matrices
        if t_adj_hyp is None:
            # If no adjacency matrix is provided, create one using the predefined mask
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            # Otherwise, initialize with the provided values while applying the mask
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Map the input features into hyperbolic space
        x_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c
        )

        # Apply the hyperbolic compression layer
        x = self.compress_layer(x_hyp)

        # Apply temporal hyperbolic graph convolution
        t_f, _ = self.t_hgcn((x, t_adj_hyp))
        # Compute updated temporal adjacency matrix based on distances in hyperbolic space
        new_T_adj_hyp = self.manifold.sqdistmatrix(t_f, self.c)
        new_T_adj_hyp = nn.Parameter(new_T_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)

        # Prepare input for spatial convolution by mapping transposed features into hyperbolic space
        y_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x.t(), self.c), c=self.c), c=self.c
        )

        # Apply spatial hyperbolic graph convolution
        s_f, _ = self.s_hgcn((y_hyp, s_adj_hyp))
        # Compute updated spatial adjacency matrix based on distances in hyperbolic space
        new_S_adj_hyp = self.manifold.sqdistmatrix(s_f, self.c)
        new_S_adj_hyp = nn.Parameter(new_S_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Combine temporal and spatial features via matrix multiplication
        x = torch.mm(t_f, s_f.t())

        # Pass the combined features through the output hyperbolic layer
        out = self.out_layer(x)

        # Project the output from hyperbolic space back to tangent (Euclidean) space
        out = self.manifold.proj_tan0(self.manifold.logmap0(out, c=self.c), c=self.c).to(device)

        # Concatenate the natural hidden (encoded) representation with the processed output
        out = torch.cat((hidden, out), dim=1)

        # Decode the combined features to reconstruct the natural input space
        out = self.decoder(out.to(torch.float64))

        # Return the final output and various auxiliary tensors
        return (
            out.view(-1),  # Flattened final output tensor
            new_T_adj_hyp.view(self.n_window, -1),  # Updated temporal adjacency matrix
            new_S_adj_hyp.view(self.feature_out_layer_dime, -1),  # Updated spatial adjacency matrix
            torch.stack([  # Stack all curvature parameters used in the model
                self.c, self.curv_liner, self.curv_t_hgcn_in, self.curv_s_hgcn_in,
                self.curv_t_hgcn_out, self.curv_s_hgcn_out, self.curv_out
            ]),
            t_adj_hyp,  # Natural temporal adjacency matrix
            s_adj_hyp   # Natural spatial adjacency matrix
        )
    
class HAO_H(nn.Module):
    def __init__(self, feats, n_windows):
        """
        Initialize the HAO_H model.

        Sets up the neural network architecture including encoder, hyperbolic graph convolution layers,
        and decoder components used during the forward pass for feature processing and reconstruction.

        The HAO_H variant specifically uses the **Hyperboloid** manifold for its geometric space,
        which allows modeling of hierarchical or complex relational data in a non-Euclidean space.

        Parameters:
            feats (int): Number of input features (feature dimensionality).
            n_windows (int): Number of time windows, which defines the size of the temporal adjacency matrix.
        """

        super(HAO_H, self).__init__()

        # Basic model identity and training parameters
        self.name = 'HAO_H'  # Model identifier
        self.lr = 0.002  # Learning rate for optimizer
        self.beta = 0.01  # Regularization coefficient

        # Feature and dimensional configuration
        self.n_feats = feats  # Input feature dimension
        self.n_hidden = 32  # Size of hidden layer in encoder/decoder
        self.n_latent = 8  # Latent space representation size
        self.n_window = n_windows  # Number of time windows for temporal graph
        self.n_latent = self.n_feats  # Latent dimension matches input feature count
        self.compress_dim = 8 * 2  # Dimension after compression in hyperbolic layers

        class argsR():
            """
            Internal configuration class to store model hyperparameters and settings.

            Parameters:
                feat_dim (int): Input feature dimension.

            Attributes:
                manifold (str): Type of geometric space ("Euclidean", "Hyperboloid", or "PoincareBall").
                num_layers (int): Number of network layers.
                act (str): Activation function name.
                feat_dim (int): Feature dimension.
                dim (int): Embedding dimension.
                n_classes (int): Number of output classes (same as input features for reconstruction).
                cuda (int): CUDA device index (-1 for CPU).
                device (str): Device type ('cuda:x' or 'cpu').
                model (str): Model type ("HGCN").
                c (float): Curvature parameter.
                task (str): Task type ("rec" for reconstruction).
                dropout (float): Dropout probability.
                bias (bool): Whether to use bias in layers.
                use_att (bool): Whether to use attention mechanism.
                local_agg (bool): Whether to use local aggregation.
                n_heads (int): Number of attention heads.
            """
            def __init__(self, feat_dim):
                self.manifold = 2
                self.num_layers = 2
                self.act = 'relu'
                self.feat_dim = feat_dim
                self.dim = feat_dim  
                self.n_classes = feat_dim  
                self.cuda = 0
                self.device = 'cuda:' + str(self.cuda) if int(self.cuda) >= 0 else 'cpu'
                self.manifold = "Hyperboloid"  # [Euclidean, Hyperboloid, PoincareBall]
                self.model = "HGCN"  
                self.c = 1.0  
                self.task = "rec"  
                self.dropout = 0.0
                self.bias = 1
                self.use_att = 1
                self.local_agg = 1
                self.n_heads = 4

        argsR = argsR(self.n_feats)
        self.argsR = argsR  # Store configuration object for later reference

        # Curvature setup
        if argsR.c is not None:
            self.c = torch.tensor([argsR.c])
            if not argsR.cuda == -1:
                self.c = self.c.to(argsR.device)  # Move curvature tensor to GPU if available
        else:
            self.c = nn.Parameter(torch.Tensor([1.0]))  # Learnable curvature parameter

        # Manifold selection based on configuration
        self.manifold = getattr(mainfolds, argsR.manifold)()  # Instantiate specified geometric space (Hyperboloid)

        # Retrieve dimensions, activation functions, and curvatures from helper module
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(argsR)

        act = acts[0]  # Use the first activation function from the list
        self.feature_out_layer_dime = self.n_feats  # Output feature dimension matches input

        # Curvature parameters for different parts of the model
        self.c = nn.Parameter(torch.Tensor([1.0]))
        self.curv_liner = nn.Parameter(torch.Tensor([1.0]))
        self.curv_t_hgcn_in = nn.Parameter(torch.Tensor([1.0]))
        self.curv_t_hgcn_out = nn.Parameter(torch.Tensor([1.0]))
        self.curv_s_hgcn_in = nn.Parameter(torch.Tensor([1.0]))
        self.curv_s_hgcn_out = nn.Parameter(torch.Tensor([1.0]))
        self.curv_out = nn.Parameter(torch.Tensor([1.0]))

        # Compression layer in hyperbolic space
        self.compress_layer = HNNLayer(self.manifold, 
                                    in_features=argsR.feat_dim * 2,
                                    out_features=self.feature_out_layer_dime,
                                    c=self.curv_liner, 
                                    act=act, 
                                    dropout=0.0, 
                                    use_bias=False).to(device)

        # Temporal hyperbolic graph convolution layer
        self.t_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, 
                                                in_features=self.feature_out_layer_dime,
                                                adj_dim=self.n_window, 
                                                adj_act=nn.Sigmoid(),
                                                out_features=self.compress_dim, 
                                                c_in=self.curv_t_hgcn_in,
                                                c_out=self.curv_t_hgcn_out, 
                                                dropout=0.0,
                                                act=act, 
                                                use_bias=False, 
                                                use_att=0, 
                                                local_agg=0).to(device)

        # Spatial hyperbolic graph convolution layer
        self.s_hgcn = HyperbolicGraphConvolution(manifold=self.manifold, 
                                                in_features=self.n_window,
                                                adj_dim=self.feature_out_layer_dime, 
                                                adj_act=nn.Sigmoid(),
                                                out_features=self.compress_dim, 
                                                c_in=self.curv_s_hgcn_in,
                                                c_out=self.curv_s_hgcn_out, 
                                                dropout=0.0,
                                                act=act, 
                                                use_bias=False, 
                                                use_att=0, 
                                                local_agg=0).to(device)

        # Output hyperbolic neural network layer
        self.out_layer = HNNLayer(self.manifold,
                                in_features=self.n_feats,
                                out_features=self.n_feats,
                                c=self.curv_out, 
                                act=act, 
                                dropout=0.0, 
                                use_bias=False).to(device)

        # Fully connected network block
        self.fcn_all = nn.Sequential(
            nn.Linear(self.n_feats * 3, 2 * self.n_latent), 
            nn.SiLU(True),
        ).to(device)

        # Adjacency masks for temporal and spatial graphs
        self.t_mask = torch.triu(torch.ones(self.n_window, self.n_window), diagonal=1).to(device) + torch.eye(
            self.n_window).to(device)
        self.s_mask = torch.triu(torch.ones(self.feature_out_layer_dime, self.feature_out_layer_dime), diagonal=1).to(
            device) + torch.eye(self.feature_out_layer_dime).to(device)

        # Encoder network: maps input into latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), 
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        ).to(device)

        # Decoder network: reconstructs original input from processed features
        self.decoder = nn.Sequential(
            nn.Linear(3 * self.n_latent, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), 
            nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), 
            nn.ReLU(),
        ).to(device)

        print(f"{self.name} Init Success")  # Notify successful initialization

    def forward(self, x, t_adj_hyp=None, s_adj_hyp=None):
        """
        Forward propagation function for the HAO_H model.

        This function performs the forward pass through the network:
        - Encodes input features into a latent representation.
        - Maps features into hyperbolic space (Hyperboloid manifold).
        - Applies temporal and spatial hyperbolic graph convolutions.
        - Combines features and decodes them to reconstruct the original input.

        Args:
            x (torch.Tensor): Input feature tensor with shape (batch_size, n_feats).
            t_adj_hyp (torch.Tensor, optional): Temporal adjacency matrix in hyperbolic space. Defaults to None.
            s_adj_hyp (torch.Tensor, optional): Spatial adjacency matrix in hyperbolic space. Defaults to None.

        Returns:
            torch.Tensor: Final output tensor after decoding (flattened).
            torch.Tensor: Updated temporal adjacency matrix.
            torch.Tensor: Updated spatial adjacency matrix.
            torch.Tensor: Stacked curvature parameters used throughout the model.
            torch.Tensor: Natural  temporal adjacency matrix.
            torch.Tensor: Natural  spatial adjacency matrix.
        """

        # Encode the input using the encoder
        x = self.encoder(x.view(-1, self.n_feats)).to(torch.float64)
        hidden = x  # Store the encoded features for later use

        # Initialize or reuse temporal and spatial adjacency matrices
        if t_adj_hyp is None:
            # If not provided, create new adjacency matrices using masks
            t_adj_hyp = nn.Parameter(self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(self.s_mask, requires_grad=True).to(device).to(torch.float64)
        else:
            # Otherwise, re-initialize with provided values and apply mask
            t_adj_hyp = nn.Parameter(t_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)
            s_adj_hyp = nn.Parameter(s_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Map input features into hyperbolic space
        x_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c
        )

        # Apply compression layer in hyperbolic space
        x = self.compress_layer(x_hyp)

        # Apply temporal hyperbolic graph convolution
        t_f, _ = self.t_hgcn((x, t_adj_hyp))
        # Update temporal adjacency matrix based on distances in hyperbolic space
        new_T_adj_hyp = self.manifold.sqdistmatrix(t_f, self.c)
        new_T_adj_hyp = nn.Parameter(new_T_adj_hyp * self.t_mask, requires_grad=True).to(device).to(torch.float64)

        # Prepare transposed input for spatial convolution
        y_hyp = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(x.t(), self.c), c=self.c), c=self.c
        )

        # Apply spatial hyperbolic graph convolution
        s_f, _ = self.s_hgcn((y_hyp, s_adj_hyp))
        # Update spatial adjacency matrix based on distances in hyperbolic space
        new_S_adj_hyp = self.manifold.sqdistmatrix(s_f, self.c)
        new_S_adj_hyp = nn.Parameter(new_S_adj_hyp * self.s_mask, requires_grad=True).to(device).to(torch.float64)

        # Combine temporal and spatial features via matrix multiplication
        x = torch.mm(t_f, s_f.t())

        # Pass through the output hyperbolic layer
        out = self.out_layer(x)

        # Project the output back from hyperbolic space to Euclidean (tangent) space
        out = self.manifold.proj_tan0(self.manifold.logmap0(out, c=self.c), c=self.c).to(device)

        # Concatenate the natural encoded (hidden) features with processed output
        out = torch.cat((hidden, out), dim=1)

        # Decode the combined features to reconstruct input space
        out = self.decoder(out.to(torch.float64))

        # Return final output and auxiliary tensors
        return (
            out.view(-1),  # Flattened output tensor
            new_T_adj_hyp.view(self.n_window, -1),  # Reshaped updated temporal adjacency matrix
            new_S_adj_hyp.view(self.feature_out_layer_dime, -1),  # Reshaped updated spatial adjacency matrix
            torch.stack([  # Stack all curvature parameters used in layers
                self.c, self.curv_liner, self.curv_t_hgcn_in, self.curv_s_hgcn_in,
                self.curv_t_hgcn_out, self.curv_s_hgcn_out, self.curv_out
            ]),
            t_adj_hyp,  # Natural temporal adjacency matrix
            s_adj_hyp   # Natural spatial adjacency matrix
    )