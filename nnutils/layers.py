# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import numpy as np
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import absl.flags as flags
FLAGS = flags.FLAGS

# ===========================================================================================================
# voxel render utils
# ===========================================================================================================
def occupancy_to_prob(voxel: torch.Tensor):
    """
    :param voxel: occupancy probability (N, 1, D, H, W)
    :return: ray stopping probability along D-axis (N, 1, D+1, H, W)
    """
    N, _, D, H, W = voxel.size()
    device = voxel.device
    safe_voxel = voxel.clamp(0, 1)

    vacancy = 1 - safe_voxel
    occupancy = safe_voxel

    if FLAGS.cum_occ == 'prod':
        ones = torch.ones([N, 1, 1, H, W], device=device)
        prev_vacancy = torch.cumprod(vacancy, dim=2)
        prev_vacancy = torch.cat([ones, prev_vacancy], dim=2)
        curr_occupancy = torch.cat([occupancy, ones], dim=2)  # last depth is background
        hit_prob = prev_vacancy * curr_occupancy
    elif FLAGS.cum_occ == 'log':
        eps = 1e-8
        zeros = torch.zeros([N, 1, 1, H, W], device=device)
        vacancy = torch.log(vacancy.clamp(min=eps))
        occupancy = torch.log(vacancy.clamp(min=eps))

        prev_vacancy = torch.cumsum(vacancy, dim=2)
        prev_vacancy = torch.cat([zeros, prev_vacancy], dim=2)
        curr_occupancy = torch.cat([occupancy, zeros], dim=2)
        hit_prob = torch.exp(prev_vacancy + curr_occupancy)
    else:
        raise NotImplementedError
    return hit_prob


def expected_wrt_prob(value, prob, miss=0.):
    """
    :param value: (N, C, D, H, W) or float?
    :param prob:  (N, 1, D + 1, H, W)
    :param miss:
    :return: (N, C, H, W)
    """
    N, _, D, H, W = prob.size()
    D -= 1

    fg_prob, bg_prob = torch.split(prob, [D, 1], dim=2)
    exp = torch.sum(fg_prob * value + bg_prob * miss, dim=2)  # N, 1, D, H, W /  N, C, D, H, W
    return exp


def expected_wrt_occupancy(value, occupancy, miss=0.):
    """
    :param value: (N, C, D, H, W) / float
    :param occupancy: (N, C, D, H, W)
    :param miss: float / (N, C, D, H, W)
    :return:
    """
    return expected_wrt_prob(value, occupancy_to_prob(occupancy), miss)


# ===========================================================================================================
# surface normal utils
# ===========================================================================================================
def grad_occ(voxel, normed=False, eps=1e-7):
    """
    :param voxel: (N, 1, D, H, W)  / zyx
    :return: unnormed (N, 3, D, H, W)
    """
    N, _, D, H, W = voxel.size()
    device = voxel.device

    voxel = voxel.squeeze(1)

    ngb_tblrfb = torch.zeros([6, N, D, H, W], device=device)
    ngb_tblrfb[0, :, :, 1:, :] = voxel[:, :, :-1, :]
    ngb_tblrfb[1, :, :, :-1, :] = voxel[:, :, 1:, :]

    ngb_tblrfb[2, :, :, :, 1:] = voxel[:, :, :, :-1]
    ngb_tblrfb[3, :, :, :, :-1] = voxel[:, :, :, 1:]

    ngb_tblrfb[4, :, 1:, :, :] = voxel[:, :-1, :, :]
    ngb_tblrfb[5, :, :-1, :, :] = voxel[:, 1:, :, :]

    # negative gradient.
    # todo. +ve direction consistency w.r.t GT
    dy = ngb_tblrfb[0] - ngb_tblrfb[1]   # (N, D, H, W)
    dx = ngb_tblrfb[2] - ngb_tblrfb[3]
    dz = ngb_tblrfb[4] - ngb_tblrfb[5]
    dz = -dz

    grad = torch.stack([dx, dy, dz], dim=1)
    if normed:
        grad = grad / (grad.norm(dim=1, keepdim=True).clamp(min=eps))
    return grad


# ===========================================================================================================
# build functions
# ===========================================================================================================
def dis_block(num_layers, dims, norm='batch', last_relu=True, k=3, d=2, p=1):
    assert num_layers + 1 == len(dims)
    layer_list = []
    for i in range(num_layers):
        layers = [conv2d(dims[i], dims[i + 1], k=k, d=d, p=p),]
        if i < num_layers - 1 or last_relu:
            layers.extend([get_norm_layer(norm, dims[i + 1]), nn.LeakyReLU(0.2, inplace=True)])
        layer_list.append(nn.Sequential(*layers))
    # return nn.Sequential(*layer_list)
    return layer_list


def dis_block_mean_std(num_layers, dims, conv='conv'):
    """norm == myin"""
    layer_list = []
    for i in range(num_layers):
        layer_list.append(ConvMeanVarBlock(
            dims[i], dims[i + 1], k=3, d=2, p=1, norm='myin', relu='leaky', conv=conv)
        )
    return layer_list


def build_gblock(d_, num_layers, dims, adain=False, z_dim=None, last_relu=True,
                 k=3, d=1, p=0, op=0, relu='relu', norm='none'):
    assert num_layers + 1 == len(dims)

    layers = nn.ModuleList()
    for l in range(num_layers):
        add_relu = (last_relu or l < num_layers - 1)
        if d_ == 3:
            conv = deconv3d
        elif d_ == 2:
            conv = deconv2d
        else:
            raise NotImplementedError
        if adain:
            layers.append(
                AdaBlock(conv, dims[l], dims[l + 1], z_dim, relu=relu, k=k, d=d, p=p, op=op))
        else:
            layers.append(
                GConvBlock(conv, dims[l], dims[l + 1],
                           add_relu=add_relu, relu=relu, norm=norm, k=k, d=d, p=p, op=op))
    return layers


def linear_block(dim_list, relu='leaky', last_relu=False, layer='regular', **kwargs):
    """
    Linear --> (relu)
    :param dim_list:
    :param relu:
    :param last_relu:
    :return:
    """
    layers = []
    for i in range(len(dim_list) - 1):
        if layer == 'regular':
            layer = linear
        elif layer == 'equal':
            layer = EqualizedLinear
        layers.append(layer(dim_list[i], dim_list[i + 1], **kwargs))
        if i < len(dim_list) - 2 or last_relu:
            layers.append(get_relu_layer(relu))
    return nn.Sequential(*layers)


def conv_block(dim_list, last_relu=False, k=3, d=1, p=0, op=0, relu='relu', norm='none'):
    num_layers = len(dim_list) - 1
    layers = []
    for i in range(num_layers):
        layers.append(conv2d(dim_list[i], dim_list[i + 1], k, d, p))
        layers.append(get_norm_layer(norm, dim_list[i + 1]))
        if i < len(dim_list) - 2 or last_relu:
            layers.append(get_relu_layer(relu))
    return nn.Sequential(*layers)

# ===========================================================================================================
# Block
# ===========================================================================================================
class AdaBlock(nn.Module):
    """
    x = adablock(z, x)
    1. x = conv(x)
    2. s, b = z_mapping(z)
    3. x = relu(adain(x, s, b))
    """

    def __init__(self, conv_func, inp_dim, out_dim, z_dim, add_conv=True,
                 relu='leaky', k=3, d=1, p=0, op=0):
        super().__init__()
        self.out_dim = out_dim

        self.conv = conv_func(inp_dim, out_dim, k, d, p, op) if add_conv else None
        if FLAGS.z_map == 'regular':
            # just an affine
            self.z_mapping = linear_block([z_dim, out_dim * 2], 'relu', last_relu=True)
        else:
            # todo：
            self.z_mapping = linear_block([z_dim, 512, out_dim * 2], 'relu', last_relu=True, layer='equal')
        self.relu = get_relu_layer(relu)

    def forward(self, *input, **kwargs):
        z, x = input
        if self.conv is not None:
            x = self.conv(x)
        s, b = torch.split(self.z_mapping(z), [self.out_dim, self.out_dim], dim=-1)
        x = AdaIn(x, s, b)
        x = self.relu(x)
        return x


class GConvBlock(nn.Module):
    """
    conv -> norm -> relu
    """

    def __init__(self, conv_func, inp_dim, out_dim, add_relu=True, relu='relu', norm='none', k=3, d=1, p=1, op=0):
        super().__init__()
        layers = [conv_func(inp_dim, out_dim, k, d, p, op)]
        layers.append(get_norm_layer(norm, out_dim))
        if add_relu:
            layers.append(get_relu_layer(relu))
        self.net = nn.Sequential(*layers)

    def forward(self, *input, **kwargs):
        x, = input
        x = self.net(x)
        return x


class ConvMeanVarBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, d=2, p=1, norm='myin', relu='leaky', conv='conv'):
        super().__init__()
        if conv == 'conv':
            self.conv = conv2d(inp_dim, out_dim, k=k, d=d, p=p)
        elif conv == 'spec':
            self.conv = nn.utils.spectral_norm(conv2d(inp_dim, out_dim, k=k, d=d, p=p))
        self.norm = MyInstanceNorm(out_dim, return_mean=True)
        self.relu = get_relu_layer(relu)

    def forward(self, *input, **kwargs):
        x, = input
        x  = self.conv(x)
        x, mean, std = self.norm(x)
        x = self.relu(x)
        return x, mean, std

# ===========================================================================================================
# Activation functions
# ===========================================================================================================

def get_relu_layer(relu='relu'):
    if relu == 'relu':
        relu_layer = nn.ReLU(inplace=True)
    elif relu == 'leaky':
        relu_layer = nn.LeakyReLU(0.2, inplace=True)
    elif relu == 'none':
        relu_layer = Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % relu)
    return relu_layer


# ===========================================================================================================
# Normalization
# ===========================================================================================================
def get_norm_layer(norm_type, dim=None):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
        if dim is not None:
            norm_layer = nn.BatchNorm2d(dim, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
        if dim is not None:
            norm_layer = nn.InstanceNorm2d(dim, affine=True, track_running_stats=False)
            nn.init.normal_(norm_layer.weight, mean=1.0, std=0.02)
            nn.init.constant_(norm_layer.bias, 0)
    elif norm_type == 'in1d':
        norm_layer = nn.InstanceNorm1d(dim, affine=True, track_running_stats=False)
        nn.init.normal_(norm_layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(norm_layer.bias, 0)
    elif norm_type == 'in3d':
        norm_layer = nn.InstanceNorm3d(dim, affine=True, track_running_stats=False)
        nn.init.normal_(norm_layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(norm_layer.bias, 0)
    elif norm_type == 'myin':
        norm_layer = MyInstanceNorm(dim)
    elif norm_type == 'none':
        norm_layer = Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class MyInstanceNorm(nn.Module):
    def __init__(self, dim, return_mean=False):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, dim, 1, 1) * 0.02 + 1)
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.return_mean = return_mean

    def forward(self, *input, **kwargs):
        """
        :param input: (N, C, H, W)
        :param kwargs:
        :return:
        """
        x, = input
        size = x.size()
        mean, std = calc_mean_std(x, 1e-5)

        x = (x - mean.expand(size)) / std.expand(size)

        x = x * self.scale + self.bias
        if self.return_mean:
            return x, mean, std
        else:
            return x


def AdaIn(features, scale, bias):
    """    Adaptive instance normalization component. Works with both 4D and 5D tensors """
    size = features.size()
    content_mean, content_std = calc_mean_std(features)
    normalized_feat = (features - content_mean.expand(size)) / content_std.expand(size)
    N, C = size[0:2]
    view_size = (N, C,) + (1,) * (len(size) - 2)
    scale = scale.view(*view_size)
    bias = bias.view(*view_size)
    normalized = normalized_feat * scale.expand(size) + bias.expand(size)
    return normalized


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4) or (len(size) == 5)
    N, C = size[0:2]
    feat_var = feat.view(N, C, -1).var(unbiased=False, dim=2) + eps
    view_size = (N, C,) + (1,) * (len(size) - 2)
    feat_std = feat_var.sqrt().view(*view_size)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(*view_size)
    return feat_mean, feat_std


def l2_norm(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)


# ===========================================================================================================
# Convolutions / linear
# ===========================================================================================================
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def conv2d(input_dim, output_dim, k=5, d=2, p=0, init='default'):
    conv = nn.Conv2d(input_dim, output_dim, k, stride=d, padding=p)
    if init == 'kaiming':
        nn.init.kaiming_normal_(conv.weight)
    elif init == 'default':
        nn.init.normal_(conv.weight, std=0.02)
        nn.init.normal_(conv.bias, 0.)
    return conv


def conv3d(inp_dim, out_dim, k=5, d=2, p=0, init='default'):
    layer = nn.Conv3d(inp_dim, out_dim, k, d, p)
    if init == 'kaiming':
        nn.init.kaiming_normal_(layer.weight)
    elif init == 'default':
        nn.init.normal_(layer.weight, std=0.02)
        nn.init.normal_(layer.bias, 0.)
    return layer


def deconv2d(inp_dim, out_dim, k=5, d=2, p=0, op=0, init='default'):
    layer = nn.ConvTranspose2d(inp_dim, out_dim, k, d, p, op)
    if init == 'kaiming':
        nn.init.kaiming_normal_(layer.weight)
    elif init == 'default':
        nn.init.normal_(layer.weight, std=0.02)
        nn.init.constant_(layer.bias, 0.)
    return layer


def deconv3d(inp_dim, out_dim, k, d, p=0, op=0, init='default'):
    layer = nn.ConvTranspose3d(inp_dim, out_dim, k, d, p, op)
    if init == 'kaiming':
        nn.init.kaiming_normal_(layer.weight)
    elif init == 'default':
        nn.init.normal_(layer.weight, std=0.02)
        nn.init.constant_(layer.bias, 0.)
    return layer


class Identity(nn.Module):
    def forward(self, x):
        return x


def linear(input_dim, output_dim, init='default', std=0.02):
    layer = nn.Linear(input_dim, output_dim)
    if init == 'kaiming':
        nn.init.kaiming_normal_(layer.weight)
    elif init == 'default':
        nn.init.normal_(layer.weight, std=std)
        nn.init.constant_(layer.bias, 0.)
    return layer


# ### StyleGAN.pytorch ####

class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class EqualizedConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** 0.5, use_wscale=False,
                 lrmul=1, bias=True, intermediate=None, upscale=False, downscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, [1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_dim, output_dim, gain=2 ** 0.5, use_wscale=True, lrmul=1e-2, bias=True):
        super().__init__()
        he_std = gain * input_dim ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_dim))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):
        """
        Mapping network used in the StyleGAN paper.
        :param latent_size: Latent vector(Z) dimensionality.
        # :param label_size: Label dimensionality, 0 if no labels.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param dlatent_broadcast: Output disentangled latent (W) as [minibatch, dlatent_size]
                                  or [minibatch, dlatent_broadcast, dlatent_size].
        :param mapping_layers: Number of mapping layers.
        :param mapping_fmaps: Number of activations in the mapping layers.
        :param mapping_lrmul: Learning rate multiplier for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param use_wscale: Enable equalized learning rate?
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[mapping_nonlinearity]

        # Embed labels and concatenate them with latents.

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(('pixel_norm', PixelNormLayer()))

        # Mapping layers. (apply_bias?)
        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps,
                                                 gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x

