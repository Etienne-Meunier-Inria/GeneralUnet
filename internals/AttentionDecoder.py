import torch.nn as nn
import torch, einops
from argparse import ArgumentParser
from .utils.PositionalEmbedding import PositionalEmbedding
from .utils.SlotAttention import SlotAttention
from .utils.TransformerAttention import TransformerAttention
from .utils.Queries import Queries


Attentions = {'transformer' : TransformerAttention, 'slot' : SlotAttention}

class AttentionDecoder(nn.Module) :

    def __init__(self, num_classes, queries_dim, perpixel_dim, queries_type='embeddings', attention='transformer', positional='Learnable', num_layers=5) :
        """
        Instanciate a transformer decoder that refine queries based on the bottleneck
        features and then compute the attention with per-pixel embedding.
        Args :
            num_classes : number of queries / number of classes in final segmentation
            queries_dim : dimension of the image features / queries.
            perpixel_dim : dimension of the embedding map used for attention with queries
            positional (str) : type of positional_embedding to use. Options : (None, Learnable, Sinusoidal)
            nhead, num_layers : parameters of the transformer block.
        """
        super().__init__()
        self.num_classes = num_classes
        self.queries_dim = queries_dim
        self.attention = attention
        self.queries_type = queries_type
        self.add_positional_embedding = PositionalEmbedding(positional, queries_dim)
        self.decoder = Attentions[attention](queries_dim, num_layers)
        self.queries = Queries(queries_type, num_classes, queries_dim)
        self.mlp = torch.nn.Linear(queries_dim, perpixel_dim)

    def get_refine_queries(self, memory_features, num_classes) :
        """
        Get queries for each class and refine them using the transformer decoder
        and features.
        Args :
            memory_features + pos (b, c, T'(opt), I', J'): features to refine the queries as "memory" in the transformer decoder
        Returns :
            queries (b k c) : queries ready to compute attention.
        """
        memory_features = einops.rearrange(memory_features, 'b c ... -> b (...) c')
        num_classes = num_classes if num_classes is not None else self.num_classes
        q = self.queries.get_init_queries(memory_features.shape[0], num_classes).to(memory_features.device)
        out_queries = self.decoder(q, memory_features)
        return out_queries


    def get_mask_embedding(self, out_queries) :
        """
        Get mask embedding from memory_features using initial queries and the mlp
        Args :
            queries (b k c) : queries ready to compute attention.
        Returns :
            mask_embeddings (b k c_maskembeding) : mask embedding for attention cross product
        """

        mask_embeddings = self.mlp(out_queries)
        return mask_embeddings


    @staticmethod
    def cross_attention(mask_embeddings, perpixel_embedding) :
        """
        Cross attention between queries and perpixel_embedding
        """
        return torch.einsum('bqf,bf...->bq...', mask_embeddings, perpixel_embedding)


    def forward(self, memory_features, perpixel_embedding) :
        """
        Compute queries using memory_features and attention map between queries
        and perpixel_embedding.
        Args :
            memory_features (b c, T'(opt), I', J') : features to refine queries
            perpixel_embedding (b ft, T(opt), I, J) : embedding extracted by the CNN, used for attention
        Returns:
            out : result of attention query-embedding ( b, L, T(opt), I, J)
        """
        memory_features = self.add_positional_embedding(memory_features)
        out_queries = self.get_refine_queries(memory_features)
        mask_embeddings = self.get_mask_embedding(out_queries)
        return self.cross_attention(mask_embeddings, perpixel_embedding)



    @staticmethod
    def add_specific_args(parser, prefix=''):
        #parser = ArgumentParser(parents=[parent_parser], add_help=False)
        prefix += 'transformer.'
        parser.add_argument(f'--{prefix}positional', help='Positional embedding technique to use', type=str,
                            choices=['Learnable', 'Sinusoidal'], default='Learnable')
        parser.add_argument(f'--{prefix}queries_type', help='The type of queries given to the decoder.', type=str,
                            choices=['gaussian', 'embeddings'], default='embeddings')
        parser.add_argument(f'--{prefix}attention', help='The type of attention process applied to the queries and input features.', type=str,
                            choices=['transformer', 'slot'], default='transformer')
        return parser
