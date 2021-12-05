import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
import os
# Temporal_LEN = 5
from global_var import *
class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True,INN_init = True,is_res = False):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)
        else:
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4,self.conv5], 1)
        self.is_res = is_res
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.is_res:
            x5 = x5+x
        return x5


class DenseBlockVideoInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True,is_res = False):
        super(DenseBlockVideoInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (1,3,3), 1, (0,1,1), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.is_res = is_res
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.is_res and x5.size() == x.size():
            return x+x5
        else:
            return x5


# class D2DTInput(nn.Module):
#     def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
#         super(D2DTInput, self).__init__()
#         self.conv1 = nn.Conv3d(channel_in, gc, (3,3,3), 1, (0,1,1), bias=bias)
#         self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
#         self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
#         self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
#         self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,3,3), 1, (1,0,0), bias=bias)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         if init == 'xavier':
#             mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
#         else:
#             mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
#         mutil.initialize_weights(self.conv5, 0)

#     def forward(self, x,io_type="2d"):
#         if io_type == "2d":
#             bt,c,w,h = x.size()
#             t = GlobalVar.get_Temporal_LEN()
#             b = bt//t
#             x  = x.reshape(b,t,c,w,h).transpose(1,2)
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         if io_type == "2d":
#             x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
#         return x5



class D2DTInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier',\
         gc=32, bias=True,INN_init = True,is_res = False):
        super(D2DTInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        if not io_type == '3d':
            io_type = "2d"
        if io_type == "2d":
            bt,c,w,h = x.size()
            # print(x.size())
            t = GlobalVar.get_Temporal_LEN()
            # t = 5
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        return x5
class D2DLTInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True,INN_init = True):
        super(D2DLTInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.early_3d_layer = nn.Conv3d(gc, gc,3, 1, 1, bias=bias)
        mutil.initialize_weights(self.early_3d_layer, 0)

        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        if not io_type == '3d':
            io_type = "2d"
        if io_type == "2d":
            bt,c,w,h = x.size()
            t = GlobalVar.get_Temporal_LEN()
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        
        x1 = self.lrelu(self.conv1(x))
        x1 = x1 + self.early_3d_layer(x1)
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        return x5
class ResD2DTInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True,INN_init = True):
        super(ResD2DTInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)
        else:
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 1)

    def forward(self, x,io_type="2d"):
        if not io_type == '3d':
            io_type = "2d"
        if io_type == "2d":
            bt,c,w,h = x.size()
            t = GlobalVar.get_Temporal_LEN()
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        out = x + x5
        if io_type == "2d":
            out = out.transpose(1,2).reshape(bt,-1,w,h)
        
        return out

class D2DInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True,INN_init = True):
        super(D2DInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (1,3,3), 1, (0,1,1), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        if io_type == "2d":
            bt,c,w,h = x.size()
            t = GlobalVar.get_Temporal_LEN()
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        return x5



class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size in {2, 4}, "Space2Depth only supports blocks size = 4 or 2"
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.block_size
        x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
        return x

    def extra_repr(self):
        return f"block_size={self.block_size}"

# import torch.nn as nn
# nn.PixelShuffle(scale)

# class PixelUnshuffle(nn.Module):
#     def __init__(self, scale=4):
#         super().__init__()
#         self.scale = scale

#     def forward(self, x):
#         N, C, H, W = x.size()
#         S = self.scale
#         # (N, C, H//bs, bs, W//bs, bs)
#         x = x.view(N, C, H // S, S, W // S, S)  
#         # (N, bs, bs, C, H//bs, W//bs)
#         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  
#         # (N, C*bs^2, H//bs, W//bs)
#         x = x.view(N, C * S * S, H // S, W // S)  
#         return x



class FeatureCalapseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 4,init='xavier', gc=32, bias=True,INN_init = True,is_res = False):
        super(FeatureCalapseBlock, self).__init__()
        self.scale = scale
        self.is_res = is_res
        if scale>1:
            self.ds = SpaceToDepth(scale)
            self.us = nn.PixelShuffle(scale)
        channel_in = (scale**2)*channel_in
        channel_out = (scale**2)*channel_out
        gc = (scale)*gc
        self.conv1 = nn.Conv3d(channel_in, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,3,3), 1, (1,1,1), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        res = x
        if self.scale>1:
            x = self.ds(x)
        if io_type == "2d":
            bt,c,w,h = x.size()
            t = GlobalVar.get_Temporal_LEN()
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        if self.scale>1:
            x5 = self.us(x5)
        if self.is_res:
            x5 = x5+res
        return x5
class FeatureCalapseBlock2D(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True,INN_init = True):
        super(FeatureCalapseBlock2D, self).__init__()
        scale = 4
        self.ds = SpaceToDepth(scale)
        self.us = nn.PixelShuffle(scale)
        channel_in = (scale**2)*channel_in
        channel_out = (scale**2)*channel_out
        gc = (scale)*gc
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (1,3,3), 1, (0,1,1), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if INN_init:
            if init == 'xavier':
                mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            else:
                mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        x = self.ds(x)
        if io_type == "2d":
            bt,c,w,h = x.size()
            t = GlobalVar.get_Temporal_LEN()
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        x5 = self.us(x5)
        return x5
class FeatureCalapseBlock_SmallC(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(FeatureCalapseBlock_SmallC, self).__init__()
        scale = 4
        self.ds = SpaceToDepth(scale)
        self.us = nn.PixelShuffle(scale)
        channel_in = (scale**2)*channel_in
        channel_out = (scale**2)*channel_out
        gc = (2)*gc
        self.conv1 = nn.Conv3d(channel_in, gc, (3,3,3), 1, (1,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,3,3), 1, (1,1,1), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        x = self.ds(x)
        if io_type == "2d":
            bt,c,w,h = x.size()
            t = GlobalVar.get_Temporal_LEN()
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        x5 = self.us(x5)
        return x5

class FeatureCalapseBlock_Fast(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(FeatureCalapseBlock_Fast, self).__init__()
        scale = 4
        self.ds = SpaceToDepth(scale)
        self.us = nn.PixelShuffle(scale)
        channel_in = (scale**2)*channel_in
        channel_out = (scale**2)*channel_out
        gc = 3*gc
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x,io_type="2d"):
        x = self.ds(x)
        if io_type == "2d":
            bt,c,w,h = x.size()
            t = GlobalVar.get_Temporal_LEN()
            b = bt//t
            x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if io_type == "2d":
            x5 = x5.transpose(1,2).reshape(bt,-1,w,h)
        x5 = self.us(x5)
        return x5

class HighOrderTNet(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(HighOrderTNet, self).__init__()
        MINI_C_NUM = 16
        self.conv  = nn.Sequential(
            nn.Conv3d(channel_in,MINI_C_NUM,1,1,0,bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.down1 = nn.Conv3d(MINI_C_NUM,MINI_C_NUM*2,(1,3,3),(1,2,2),(0,1,1),bias=bias)

        self.inner1_block = D2DTInput(MINI_C_NUM*2,MINI_C_NUM*2,gc=32)

        self.down2 = nn.Conv3d(MINI_C_NUM*2,MINI_C_NUM*4,(1,3,3),(1,2,2),(0,1,1),bias=bias)

        self.inner2_block = D2DTInput(MINI_C_NUM*4,MINI_C_NUM*4,gc=32)

        self.down3 = nn.Conv3d(MINI_C_NUM*4,MINI_C_NUM*8,(1,3,3),(1,2,2),(0,1,1),bias=bias)

        self.inner3_block = D2DTInput(MINI_C_NUM*8,MINI_C_NUM*8,gc=32)


        self.up0 = nn.Sequential(
            nn.Upsample(None,(1,2,2)),
            nn.Conv3d(MINI_C_NUM*8,MINI_C_NUM*4,(1,3,3),1,(0,1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(None,(1,2,2)),
            nn.Conv3d(MINI_C_NUM*4,MINI_C_NUM*2,(1,3,3),1,(0,1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(None,(1,2,2)),
            nn.Conv3d(MINI_C_NUM*2,MINI_C_NUM,(1,3,3),1,(0,1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.tail = nn.Sequential(
            nn.Conv3d(MINI_C_NUM,channel_out,1,1,0)
        )



        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv[0], self.down1, self.down2, self.up1[1]\
                ,self.up2[1]], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.tail[0], 0)

    def forward(self, x):
        bt,c,w,h = x.size()
        t = GlobalVar.get_Temporal_LEN()
        b = bt//t
        x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.conv(x)
        x_down1 = self.down1(x1)
        x_down1_t = self.inner1_block(x_down1,io_type="3d")
        x_down2 = self.down2(x_down1_t)
        x_down2_t = self.inner2_block(x_down2,io_type="3d")
        x_down3 = self.down3(x_down2_t)
        x_down3_t = self.inner3_block(x_down3,io_type="3d")
        
        x_out = self.up0(x_down3_t) + x_down2_t

        x_out = self.up1(x_out) + x_down1_t
        x_out = self.up2(x_out) + x1
        x_out = self.tail(x_out)
        x_out = x_out.transpose(1,2).reshape(bt,-1,w,h)
        return x_out



class HighOrderTNet1(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(HighOrderTNet1, self).__init__()
        MINI_C_NUM = 16
        self.conv1  = nn.Sequential(
            nn.Conv3d(channel_in,MINI_C_NUM,1,1,0,bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.down1 = nn.Conv3d(MINI_C_NUM,MINI_C_NUM*2,(1,3,3),(1,2,2),(0,1,1),bias=bias)

        self.inner1_block = D2DTInput(MINI_C_NUM*2,MINI_C_NUM*2,gc=32)

        self.down2 = nn.Conv3d(MINI_C_NUM*2,MINI_C_NUM*4,(1,3,3),(1,2,2),(0,1,1),bias=bias)

        self.inner2_block = D2DTInput(MINI_C_NUM*4,MINI_C_NUM*4,gc=32)

        self.up1 = nn.Sequential(
            nn.Upsample(None,(1,2,2)),
            nn.Conv3d(MINI_C_NUM*4,MINI_C_NUM*2,(1,3,3),1,(0,1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(None,(1,2,2)),
            nn.Conv3d(MINI_C_NUM*2,MINI_C_NUM,(1,3,3),1,(0,1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.tail = nn.Sequential(
            nn.Conv3d(MINI_C_NUM,channel_out,1,1,0)
        )



        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1[0], self.down1, self.down2, self.up1[1]\
                ,self.up2[1]], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.tail, 0)

    def forward(self, x):
        bt,c,w,h = x.size()
        t = 3
        b = bt//t
        x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.conv1(x)
        x_down1 = self.down1(x1)
        x_down1_t = self.inner1_block(x_down1,io_type="3d")
        x_down2 = self.down2(x_down1_t)
        x_down2_t = self.inner2_block(x_down2,io_type="3d")

        x_out = self.up1(x_down2_t) + x_down1_t
        x_out = self.up2(x_out) + x1
        x_out = self.tail(x_out)
        x_out = x_out.transpose(1,2).reshape(bt,-1,w,h)
        return x_out

class HighOrderTNet1(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(HighOrderTNet1, self).__init__()
        MINI_C_NUM = 16
        self.conv1  = nn.Sequential(
            nn.Conv3d(channel_in,MINI_C_NUM,1,1,0,bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.down1 = nn.Conv3d(MINI_C_NUM,MINI_C_NUM*2,(1,3,3),(1,2,2),(0,1,1),bias=bias)

        self.inner1_block = D2DTInput(MINI_C_NUM*2,MINI_C_NUM*2,gc=64)

        self.up2 = nn.Sequential(
            nn.Upsample(None,(1,2,2)),
            nn.Conv3d(MINI_C_NUM*2,MINI_C_NUM,(1,3,3),1,(0,1,1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.tail = nn.Sequential(
            nn.Conv3d(MINI_C_NUM,channel_out,1,1,0)
        )



        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1[0], self.down1, self.up2[1]], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.tail, 0)

    def forward(self, x):
        bt,c,w,h = x.size()
        t = 3
        b = bt//t
        x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.conv1(x)
        x_down1 = self.down1(x1)
        x_down1_t = self.inner1_block(x_down1,io_type="3d")

        x_out = self.up2(x_down1_t) + x1
        x_out = self.tail(x_out)
        x_out = x_out.transpose(1,2).reshape(bt,-1,w,h)
        return x_out


class D2DTEnhanceInput(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(D2DTEnhanceInput, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv51 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (1,0,0), bias=bias)
        self.conv52 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1, (2,0,0),(2,1,1), bias=bias)
        self.conv53 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3,1,1), 1,(3,0,0),(3,1,1), bias=bias)

        self.conv6 = nn.Conv3d(channel_out*3, channel_out, (1,1,1), 1,0, bias=bias)



        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4,\
                self.conv51,self.conv52,self.conv53,self.conv6], 0.1)
        mutil.initialize_weights(self.conv6, 0)

    def forward(self, x):
        bt,c,w,h = x.size()
        t = GlobalVar.get_Temporal_LEN()
        b = bt//t
        x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        spatial_f = torch.cat((x, x1, x2, x3, x4), 1)
        x51 = self.lrelu(self.conv51(spatial_f))
        x52 = self.lrelu(self.conv52(spatial_f))
        x53 = self.lrelu(self.conv53(spatial_f))
        temproal_f = torch.cat((x51,x52,x53), 1)
        x6 = self.conv6(temproal_f)

        x6 = x6.transpose(1,2).reshape(bt,-1,w,h)
        return x6

class DenseBlock3D(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.channel_out = channel_out
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        bt,c,w,h = x.size()
        t = GlobalVar.get_Temporal_LEN()
        b = bt//t
        x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = x5.transpose(1,2).reshape(bt,self.channel_out,w,h)
        return x5
class DenseBlock3DPartial(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock3DPartial, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1,3,3), 1, (0,1,1), bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.channel_out = channel_out
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        bt,c,w,h = x.size()
        t = GlobalVar.get_Temporal_LEN()
        b = bt//t
        x  = x.reshape(b,t,c,w,h).transpose(1,2)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = x5.transpose(1,2).reshape(bt,self.channel_out,w,h)
        return x5



def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out,gc = 32):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        if net_structure == 'DB3DNet':
            if init == 'xavier':
                return DenseBlock3D(channel_in, channel_out, init)
            else:
                return DenseBlock3D(channel_in, channel_out)
        if net_structure == 'FeatureCalapseBlock':
            if init == 'xavier':
                return FeatureCalapseBlock(channel_in, channel_out, init)
            else:
                return FeatureCalapseBlock(channel_in, channel_out)
        if net_structure == 'FeatureCalapseBlock_SmallC':
            if init == 'xavier':
                return FeatureCalapseBlock_SmallC(channel_in, channel_out, init)
            else:
                return FeatureCalapseBlock_SmallC(channel_in, channel_out)
        if net_structure == 'FeatureCalapseBlock_Fast':
            if init == 'xavier':
                return FeatureCalapseBlock_Fast(channel_in, channel_out, init)
            else:
                return FeatureCalapseBlock_Fast(channel_in, channel_out)
        
        if net_structure == 'D2DTNet':
            if init == 'xavier':
                return D2DTInput(channel_in, channel_out, init,gc = gc)
            else:
                return D2DTInput(channel_in, channel_out)
        if net_structure == 'ResD2DTInput':
            if init == 'xavier':
                return ResD2DTInput(channel_in, channel_out, init,gc = gc)
            else:
                return ResD2DTInput(channel_in, channel_out)
        
        if net_structure == 'D2DNet':
            if init == 'xavier':
                return D2DInput(channel_in, channel_out, init)
            else:
                return D2DInput(channel_in, channel_out)
        if net_structure == 'D2DLTInput':
            if init == 'xavier':
                return D2DLTInput(channel_in, channel_out, init)
            else:
                return D2DLTInput(channel_in, channel_out)
        if net_structure == 'D2DTEnhanceInput':
            if init == 'xavier':
                return D2DTEnhanceInput(channel_in, channel_out, init)
            else:
                return D2DTEnhanceInput(channel_in, channel_out)
        if net_structure == 'HighOrderTNet':
            if init == 'xavier':
                return HighOrderTNet(channel_in, channel_out, init)
            else:
                return HighOrderTNet(channel_in, channel_out)

        
        if net_structure == 'DB3DNet_P':
            if init == 'xavier':
                return DenseBlock3DPartial(channel_in, channel_out, init)
            else:
                return DenseBlock3DPartial(channel_in, channel_out)
        else:
            return None

    return constructor
