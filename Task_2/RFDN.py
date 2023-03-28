import torch
import torch.nn as nn
import block as B

def make_model(args, parent=False):
    model = RFDN()
    return model


class RFDN(nn.Module):
    def __init__(self, in_nc=1, nf=50, num_modules=1, out_nc=1, upscale=1):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(4 * nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, 1, kernel_size=3)

        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        res = self.LR_conv(out_B)
        return res + input


    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx