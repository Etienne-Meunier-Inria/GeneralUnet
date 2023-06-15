import torch.nn as nn
import torch
import einops

class cInstanceNorm2d(nn.Module):

    def __init__(self, num_features, affine=False, track_running_stats=False, epsilon=1e-05) :
        """
        Applies Instance Normalisation on the 2 last dimensions of a N-d input.
        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        """
        super().__init__()
        assert track_running_stats == False, f'track_running_stats = {track_running_stats} not implemented'
        assert affine == False, f'track_running_stats = {affine} not implemented'

        self.eps = epsilon
        self.num_features = num_features

    def forward(self, x) :
        """
        Applies layer normalisation.

        Args :
            x (N, C, *, H, W) : applies layer normalisation over (H, W).
                                Each channels (N, C, *) is independent
        Returns :
            normalised x (N, C, *, H, W)
        """
        assert x.shape[1] == self.num_features,\
         f'Expected (N, {self.num_features}, ...), got {x.shape}'

        r = (x - x.mean(axis=(-2, -1), keepdim=True))
        return r / torch.sqrt(x.var(axis=(-2, -1), keepdim=True, unbiased=False) + self.eps)

class cBlockNorm3d(nn.Module):

    def __init__(self, num_features, block_size=9, affine=False, track_running_stats=False, epsilon=1e-05) :
        """
        Normalise spatio temporal block with blocks of block_size dimension
        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        """
        super().__init__()
        assert track_running_stats == False, f'track_running_stats = {track_running_stats} not implemented'
        assert affine == False, f'track_running_stats = {affine} not implemented'

        self.eps = epsilon
        self.num_features = num_features
        self.block_size = block_size

    def forward(self, x) :
        """
        Applies layer normalisation.

        Args :
            x (N, C, T, H, W) : applies layer normalisation over (block_size, H, W).
                                Each channel (N, C, *) is and each block is
        Returns :
            normalised x (N, C, T, H, W)
        """
        assert x.shape[1] == self.num_features,\
         f'Expected (N, {self.num_features}, ...), got {x.shape}'
        t = x.shape[2]
        if t % self.block_size != 0 :
            p = (self.block_size - t % self.block_size)
            x =nn.functional.pad(x, (0,0,0,0,p // 2 , p// 2+ p%2), 'reflect')
        num_groups = (x.shape[2] // self.block_size)
        x = einops.rearrange(x, 'b c t i j -> (b c) t i j', c=self.num_features)
        x = nn.functional.group_norm(x, num_groups)
        x = einops.rearrange(x, '(b c) t i j -> b c t i j', c=self.num_features)
        if t % self.block_size != 0 :
            x = x[:,:,p//2:-(p// 2+ p%2)]
        return x
