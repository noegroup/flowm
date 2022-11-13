import torch
from torch import nn


__all__ = ["calc_lipschitz_norm", "ZscoreLayer", "FCLayer"]

def calc_lipschitz_norm(w):
    """Approximate the L2-Lip coeff for Wx, which is just the spectral norm of W
        # References 
        - [Regularisation of Neural Networks by Enforcing Lipschitz Continuity]
          (https://arxiv.org/pdf/1804.04368.pdf)
    """
    with torch.no_grad():
        b_k = torch.rand(w.shape[1:], dtype=w.dtype, device=w.device)

        for _ in range(5):                                  
            b_k1 = torch.mv(w, b_k)                             
            b_k = torch.mv(w.T, b_k1)

        norm = torch.norm(torch.mv(w, b_k)) / torch.norm(b_k)
    return norm

class ZscoreLayer(nn.Module):
    """Layer for Zscore normalization. Zscore normalization involves
    scaling features by their mean and standard deviation in the following
    way:

        X_normalized = (X - X_avg) / sigma_X

    where X_normalized is the zscore-normalized feature, X is the original
    feature, X_avg is the average value of the orignal feature, and sigma_X
    is the standard deviation of the original feature.

    Parameters
    ----------
    zscores: torch.Tensor
        [2, n_features] tensor, where the first row contains the means
        and the second row contains the standard deviations of each
        feature

    Notes
    -----
    Zscore normalization can accelerate training convergence if placed
    after a GeometryFeature() layer, especially if the input features
    span different orders of magnitudes, such as the combination of angles
    and distances.

    For more information, see the documentation for
    sklearn.preprocessing.StandardScaler

    """

    def __init__(self, zscores):
        super(ZscoreLayer, self).__init__()
        self.register_buffer('zscores', zscores)

    def forward(self, in_feat):
        """Normalizes each feature by subtracting its mean and dividing by
           its standard deviation.

        Parameters
        ----------
        in_feat: torch.Tensor
            input data of shape [n_frames, n_features]

        Returns
        -------
        rescaled_feat: torch.Tensor
            Zscore normalized features. Shape [n_frames, n_features]

        """
        rescaled_feat = (in_feat - self.zscores[0, :])/self.zscores[1, :]
        return rescaled_feat

class FCLayer(nn.Module):
    def __init__(self, width, activation, out_width=None, bias=True, batch_norm=False, layer_norm=True, skip=False):
        super(FCLayer, self).__init__()
        if out_width is not None and out_width != width:
            if skip:
                raise ValueError("Not possible to use skip connection when out_width is not the same as the input.")
        else:
            out_width = width
        if batch_norm and layer_norm:
            raise ValueError("`batch_norm` and `layer_norm` cannot be both turned on.")
        self.layer = nn.Linear(width, out_width, bias=bias)
        self.act = activation
        self.skip = skip
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_width)
        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = nn.LayerNorm(out_width, elementwise_affine=False)
    
    def forward(self, x):
        y = self.layer(x)
        if self.batch_norm:
            y = self.bn(y)
        if self.layer_norm:
            y = self.ln(y)
        z = self.act(y)
        if self.skip:
            return z + x
        else:
            return z
    
    def lipschitz_projection(self, lip_strength=10.0):
        """Check the L2 Lipschitz norm of each linear layer and scale the weight matrix
        when the spectral norm exceeds the given lipschitz strength."""
        for name, par in self.layer.named_parameters():
            if name == "weight":
                norm = calc_lipschitz_norm(par.data)
                par.data /= max(1., norm / lip_strength)
