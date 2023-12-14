import torch.nn as nn
import torch, einops
from torch.nn import init
from ShapeChecker import ShapeCheck
from ipdb import set_trace
import math

class Queries(nn.Module) :
    def __init__(self, queries_type, num_queries, queries_dim) :
        super().__init__()
        if queries_type == 'gaussian' :
            self.queriesInit = QueriesGaussian(queries_dim)
        elif queries_type == 'embeddings' :
            self.queriesInit = QueriesEmbeddings(num_queries, queries_dim)
        else :
            print(f'Queries type : {queries_type} not known')

    def get_init_queries(self, batch_size, num_queries) :
        """
        Initialization of the queries.
        Args:
            batch_size
            queries_dim : dimension of the queries
            num_queries : number of queries
            device : memory_features.device
            dtype : memory_feature.dtype
        """
        return self.queriesInit.get_init_queries(batch_size, num_queries)


class QueriesGaussian(nn.Module):
    def __init__(self, queries_dim) :
        super().__init__()
        self.queries_dim = queries_dim
        self.slots_mu = nn.Parameter(torch.randn(queries_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(queries_dim))
        # Xavier init with 1 dim (gain=1.0)
        with torch.no_grad() :
            g = math.sqrt(3.0) * math.sqrt(1.0/float(queries_dim))
            self.slots_logsigma.uniform_(-g, g)

        #init.xavier_uniform_(self.slots_logsigma)

    def get_init_queries(self, batch_size, num_queries) :
        """
        Initialization of the gaussian queries.
        Args:
            batch_size
            queries_dim : dimension of the queries
            num_queries : number of queries
            device : memory_features.device
            dtype : memory_feature.dtype
        """
        sc = ShapeCheck()
        sc.update([num_queries, self.queries_dim, batch_size], 'num_queries queries_dim batch_size')

        mu = sc.repeat(self.slots_mu, 'queries_dim -> batch_size num_queries queries_dim')
        sigma = sc.repeat(self.slots_logsigma.exp(), 'queries_dim -> batch_size num_queries queries_dim')

        return mu + sigma * torch.randn(mu.shape, dtype=mu.dtype, device=mu.device)


class QueriesEmbeddings(nn.Module):
    def __init__(self, num_queries, queries_dim) :
        super().__init__()
        self.num_queries = num_queries
        self.queries = torch.nn.Embedding(num_queries, queries_dim)

    def get_init_queries(self, batch_size, num_queries) :
        """
        Initialization of the queries with the embedding layer.
        Args:
            batch_size
            queries_dim : dimension of the queries
            num_queries : number of queries
            device : memory_features.device
            dtype : memory_feature.dtype
        """
        assert self.num_queries == num_queries, "The number of queries must be equal to the number of classes."
        return einops.repeat(self.queries(torch.arange(num_queries, device=self.queries.weight.device)), 'k c -> b k c', b=batch_size)
