from .unet_clean import UNetClean
import torch.nn as nn
from .u3D_3D import Unet3D_3D
import torch, einops
from ipdb import set_trace
from argparse import ArgumentParser
from .internals.AttentionDecoder import AttentionDecoder
from torch.nn.functional import interpolate
from .utils.ArgsUtils import argf

class MaskFormer3D(Unet3D_3D) :

    def __init__(self, version ='mf1', **kwargs) :
        super().__init__(**kwargs)
        self.version = version
        print(f'Using maskformer version {self.version}')
        fbp = self.features_start * 2**(self.num_layers-1)
        self.init_modules(self.num_classes, fbp, self.features_start, **kwargs)

    def forward(self,x) :
        if self.version == 'mf1' :
            return self.forward_mf1(x)
        elif self.version == 'mf2' :
            return self.forward_mf2(x)
        elif self.version == 'mf3' :
            return self.forward_mf3(x)

    def init_modules(self, c_classes, c_bottlenecks,  c_pixel_embedding, **kwargs):
        """
        c_classes : number of final output classes
        c_bottlenecks : numer of channels in bottleneck embedding
        c_pixel_embedding : number of channels in the decoder pixel embedding
        transformer_positional : type of positional encoding to use.
        """
        self.num_classes = c_classes
        #set_trace()
        if self.version in ('mf1', 'mf2') :
            print('Init final layer as Identity')
            self.layers['final_layer'] = nn.Identity()
            self.Ta = AttentionDecoder(c_classes, c_bottlenecks, c_pixel_embedding,
                                       **argf(kwargs, 'transformer'))
        elif self.version == 'mf3' :
            self.Ta = AttentionDecoder(c_classes, c_bottlenecks, c_bottlenecks,
                                        **argf(kwargs, 'transformer'))
            self.mlp_bottleneck = nn.Sequential(nn.Linear(c_classes, c_bottlenecks),
                                               nn.ReLU())

    def forward_mf1(self, x):
        """
        Params :
            x : model input ( b, c, T, I, J)
        Returns:
            Segmentation : model segmentation ( b, L, T, I , J)
            auxs : auxiliary output dict with
                HiddenV : list with skip hidden representation ( num_layers, b, Ls, Ts, Is, Js)
                Queries : Queries computed by the transformer for this batch (b, k, c_queries)
        """
        bottleneck, skips = self.encode(x)
        perpixel_embedding = self.decode(bottleneck, skips)


        bottleneck = self.Ta.add_positional_embedding(bottleneck)
        out_queries = self.Ta.get_refine_queries(bottleneck, self.num_classes)
        mask_embeddings = self.Ta.get_mask_embedding(out_queries)

        out = self.Ta.cross_attention(mask_embeddings, perpixel_embedding)

        set_trace()
        hidden_feats = skips + [bottleneck]
        return out,  {'HiddenV' : hidden_feats,
                      'Queries' : out_queries}

    def forward_mf2(self, x):
        """
        Params :
            x : model input ( b, c, T, I, J)
        Returns:
            Segmentation : model segmentation ( b, L, T, I , J)
            auxs : auxiliary output dict with
                HiddenV : list with skip hidden representation ( num_layers, b, Ls, Ts, Is, Js)
                Queries : Queries computed by the transformer for this batch (b, k, c_queries)
        """
        bottleneck, skips = self.encode(x)
        bottleneck = self.Ta.add_positional_embedding(bottleneck)

        perpixel_embedding = self.decode(bottleneck, skips)

        out_queries = self.Ta.get_refine_queries(bottleneck, self.num_classes)
        mask_embeddings = self.Ta.get_mask_embedding(out_queries)

        out = self.Ta.cross_attention(mask_embeddings, perpixel_embedding)

        hidden_feats = skips + [bottleneck]

        return out, {'HiddenV' : hidden_feats, 'Queries' : out_queries,
                     'PerPixelEmbd' :perpixel_embedding, 'MaskEmbd' :mask_embeddings}

    def forward_mf3(self, x):
        """
        OCLR structure
        Params :
            x : model input ( b, c, T, I, J)
        Returns:
            Segmentation : model segmentation ( b, L, T, I , J)
            auxs : auxiliary output dict with
                HiddenV : list with skip hidden representation ( num_layers, b, Ls, Ts, Is, Js)
        """
        bottleneck, skips = self.encode(x)

        bottleneck = self.Ta.add_positional_embedding(bottleneck)
        out_queries = self.Ta.get_refine_queries(bottleneck, self.num_classes)
        mask_embeddings = self.Ta.get_mask_embedding(out_queries)

        bottleneck = self.Ta.cross_attention(mask_embeddings, bottleneck)
        bottleneck = einops.rearrange(bottleneck, 'b q t i j -> b t i j q')
        bottleneck = self.mlp_bottleneck(bottleneck)
        bottleneck = einops.rearrange(bottleneck, 'b t i j q -> b q t i j')
        perpixel_embedding = self.decode(bottleneck, skips)
        out = self.layers['final_layer'](perpixel_embedding)

        hidden_feats = skips + [bottleneck]
        return out, {'HiddenV' : hidden_feats, 'Queries' : out_queries}

    @staticmethod
    def add_specific_args(parser, prefix=''):
        #parser = ArgumentParser(parents=[parent_parser], add_help=False)
        prefix+='maskformer.'
        parser = AttentionDecoder.add_specific_args(parser, prefix=prefix)
        parser.add_argument(f'--{prefix}version', help='Version of MaskFormer to use', type=str,
                            choices=['mf1', 'mf2', 'mf3'], default='mf2')
        return parser
