from .unet_clean import UNetClean
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, train_bn: bool, inner_normalisation: str, padding_mode:str, **kwargs):
        super().__init__()
        inm = {'InstanceNorm':nn.InstanceNorm2d, 'BatchNorm':nn.BatchNorm2d, 'None' : nn.Identity}[inner_normalisation]
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            inm(out_ch, track_running_stats=train_bn),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
            inm(out_ch, track_running_stats=train_bn),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downsampling (by Maxpooling)
    """

    def __init__(self, in_ch: int, out_ch: int, scale_factor: int, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        assert out_ch == in_ch, f'Expected out_ch = {in_ch} in Up got out_ch = {out_ch}'
        self.dowsample = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        """Downsample input by scale factor

        Parameters
        ----------
        x : input tensor ( b, c, w, h)

        Returns
        -------
        x : output tensor (b, c, w // sf, h // sf)

        """
        assert x.shape[1] ==  self.in_ch, 'Incorrect number of channels'
        return self.dowsample(x)

class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    """

    def __init__(self, in_ch: int, out_ch: int,  scale_factor: int, padding_mode:str, bilinear : bool, **kwargs):
        super().__init__()
        assert out_ch == in_ch//scale_factor, f'Expected out_ch = {in_ch//scale_factor} in Up got out_ch = {out_ch}'
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, padding_mode=padding_mode),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=scale_factor, stride=scale_factor,
                                               padding_mode=padding_mode)

    def forward(self, x1):
        return self.upsample(x1)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--bilinear', action='store_true',
                            help='bilinear upsampling in unet instead of convtranspose')
        return parser



class Unet2D_2D(UNetClean) :
    lalevels = { 'down' : Down,
                 'convl' : DoubleConv,
                 'transit' : nn.Identity,
                 'convr' : DoubleConv,
                 'up' : Up,
                 'final_layer' : nn.Conv2d}

    def __init__(self, **kwargs) :
        super().__init__(**kwargs)


    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = Up.add_specific_args(parent_parser)
        return parser
