import torch.nn as nn
import torch, einops


class TransformerAttention(nn.Module) :

    def __init__(self, queries_dim, num_layers=5, nhead=2) :
        """
        Instanciate a transformer decoder
        Args :
            queries_dim : dimension of the image features / queries.
            nhead, num_layers : parameters of the transformer block.
        """
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=queries_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        print('Init transformer decoder using Xavier : TransformerDecoderXavier')
        self._reset_parameters()

    def forward(self, queries, memory_features):
        return self.transformer_decoder(queries, memory_features)


    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
