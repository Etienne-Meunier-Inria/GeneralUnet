import torch.nn as nn
import torch, einops

class PositionalEmbedding(nn.Module) :
    def __init__(self, positional, out_dim) :
        super().__init__()
        self.positional = positional
        if positional == 'Learnable' :
            self.proj= nn.Linear(3, out_dim)
            self.get_embedding =  self.embedding_learnable
            self.gds = {} # Grid cache.
        elif positional == 'Sinusoidal' :
            raise NotImplementedError(f'PositionalEmbedding : {positional} not implemented')
        elif positional is not None:
            raise Exception(f'PositionalEmbedding : {positional} not known')

    def get_grid(self, shape) :
        if shape not in self.gds :
            print(f'Set grid : {shape}')
            T, I , J = shape
            d = self.proj.weight.device
            t, i, j = torch.meshgrid(torch.linspace(0, 1, T, device=d),
                                     torch.linspace(0, 1, I, device=d),
                                     torch.linspace(0, 1, J, device=d), indexing='ij')
            self.gds[shape] = einops.rearrange([t, i, j], 'c t i j -> t i j c')
        return self.gds[shape]


    def embedding_learnable(self, shape) :
        """
        shape : shape of the grid to build ( t, i, j)
        Return a flat grid with the parameters :
            Embedding Grid : ( c, t, i, j), c=out_dim
        """
        gd = self.get_grid(shape)
        embedding = einops.rearrange(self.proj(gd), 't i j c -> c t i j')
        return embedding

    def forward(self, features) :
        """
        Add positional_embedding to features.
        Args :
            memory_features (b, c, T'(opt), I', J'): features to refine the queries as "memory" in the transformer decoder
        Returns :
            pos_memory_features (b, c, T'(opt), I', J') : memory_features + positional_embedding
        """
        if features.ndim == 4 :
            T, I, J = 1, features.shape[-2:]
        elif features.ndim == 5 :
            T, I, J = features.shape[-3:]
        else :
            raise Exception(f'Error in shape features : {features.shape}')
        out = features
        if self.positional is not None :
            out = out + torch.squeeze(self.get_embedding((T, I, J)))
        return out
