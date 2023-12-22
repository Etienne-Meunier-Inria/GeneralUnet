import torch.nn as nn
import torch, einops
from torch.nn import init
from argparse import ArgumentParser
from ipdb import set_trace

import sys
from ShapeChecker import ShapeCheck

# Code inspired by https://github.com/lucidrains/slot-attention


class SlotAttention(nn.Module) :

    def __init__(self, queries_dim, num_layers=3, hidden_dim=128) :
        """
        Instanciate the slot attention module, which implements the slot attention process
        Args :
            queries_dim : dimension of the queries.
            inputs_dim : dimension of the image features
            num_layers : number of iterations done during the attention process
                         name choosen for uniformisation with normal attention
                         later named "iters" in the code
        """
        super().__init__()
        self.init_slot_attention(queries_dim=queries_dim, inputs_dim=None, iters=num_layers, hidden_dim=hidden_dim)

    def init_slot_attention(self, queries_dim, inputs_dim, iters, hidden_dim, eps=1e-8):

        inputs_dim = inputs_dim if inputs_dim is not None else queries_dim

        self.iters = iters
        self.eps = eps #epsilon
        self.scale = queries_dim ** -0.5

        self.to_q = nn.Linear(queries_dim, queries_dim)
        self.to_k = nn.Linear(inputs_dim, queries_dim)
        self.to_v = nn.Linear(inputs_dim, queries_dim)

        self.gru = nn.GRUCell(queries_dim, queries_dim)

        hidden_dim = max(inputs_dim, hidden_dim, queries_dim)

        self.mlp_slot = nn.Sequential(
            nn.Linear(queries_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, queries_dim)
        )

        self.norm_input  = nn.LayerNorm(inputs_dim)
        self.norm_queries  = nn.LayerNorm(queries_dim)
        self.norm_pre_ff = nn.LayerNorm(queries_dim)


    def step(self, queries, k, v, sc) :
        q = self.to_q(self.norm_queries(queries))

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps

        attn_sum = sc.reduce(attn, 'batch_size num_queries num_samples -> batch_size num_queries ()', 'sum')
        attn = attn / attn_sum

        updates = torch.einsum('bjd,bij->bid', v, attn)

        queries = self.gru(
            sc.rearrange(updates, 'batch_size num_queries queries_dim -> (batch_size num_queries) queries_dim'),
            sc.rearrange(queries, 'batch_size num_queries queries_dim -> (batch_size num_queries) queries_dim')
        )

        queries = sc.rearrange(queries, '(batch_size num_queries) queries_dim -> batch_size num_queries queries_dim')
        queries = queries + self.mlp_slot(self.norm_pre_ff(queries))
        return queries

    def forward(self, queries, inputs):
        """
        Implements the slot attention process described in the paper "Object-centric learning with slot attention"
        Args :
            inputs (batch n_classes dim): inputs given to the Slot Attention module (will obtain keys and values from them, b = batch size, n = number of regions/masks)
            queries (batch n_queries queries_dim): queries to update during the process and return
        Returns :
            queries (batch n_queries dim)
        """
        sc = ShapeCheck()
        sc.update(inputs.shape, 'batch_size num_samples inputs_dim')
        sc.update(queries.shape, 'batch_size num_queries queries_dim')
        device, dtype = inputs.device, inputs.dtype

        inputs = self.norm_input(inputs) #2, 5376, 128
        k, v = self.to_k(inputs), self.to_v(inputs)

        sc.update(k.shape, 'batch_size num_samples queries_dim')
        sc.update(v.shape, 'batch_size num_samples queries_dim')
        sc.update(queries.shape, 'batch_size num_queries queries_dim')

        for _ in range(self.iters-1):
            queries =  self.step(queries, k, v, sc)

        queries = self.step(queries, k, v, sc)

        return queries
