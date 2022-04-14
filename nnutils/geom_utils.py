# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import pytorch3d.transforms as py_t

# ################ Rotation ################
def u6d2azel(u, ):
    """
    :param u: (N, 6)
    :return: (N, 2) nearest neighbor in az,el
    """
    rot = u6d2rot(u)
    azel = nn_rot2azel(rot)
    return azel


def nn_rot2azel(rot):
    """
    :param rot: (N, 3, 3)
    :return: (N, 2)
    """
    ea3 = rot2euler(rot)  # N, 3
    # drop the tilt
    azel, tilt = torch.split(ea3, [2, 1], -1)
    # mask = (tilt > np.pi / 2).float()
    # azel = -mask * azel+ (1 - mask) * azel  # if tilt > np.pi /2 --> flip?

    return azel


def rot2euler(rot):
    """
    :param rot: (N, 3, 3)
    :return: (N, 3)
    """
    ea3 = -py_t.matrix_to_euler_angles(rot, 'YXZ')  # negative for consitency
    return ea3



def u6d2rot(u, homo=False):
    """
    :param u: (N, 6)
    :return: (N, 3, 3)
    """
    a1, a2 = torch.split(u, [3, 3], dim=-1)
    b1 = F.normalize(a1, dim=-1)  # N, 3
    b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1, dim=-1)
    b3 = b1.cross(b2)

    rot = torch.stack([b1, b2, b3], dim=-1)
    if homo:
        rot = homo_matrix(rot)
    return rot

def rot2u6d(rot):
    """
    :param rot: (N, 3, 3)
    :return: (N, 6)
    """
    a1, a2, _ = torch.split(rot, [1, 1, 1], dim=-1)  # N, 3, 1
    u = torch.cat([a1.squeeze(-1), a2.squeeze(-1)], dim=1)
    return u

def azel2u6d(v, ):
    """
    :param v: (N, 2)
    :return: (N, 6)
    """
    rot = azel2rot(v[:, 0], v[:, 1], False)
    u = rot2u6d(rot)
    return u

def azel2uni(view_para, homo=True):
    """
    :param view_para: tensor in shape of (N, 6 / 2) az, el, scale, x, y, z
    :return: scale: (N, 1), trans: (N, 3). rot: (N, 4, 4)
    """
    if view_para.size(1) == 2:
        az, el = torch.split(view_para, [1, 1], dim=1)
        zeros = torch.zeros_like(az)
        ones = torch.ones_like(az)

        view_para = torch.cat([az, el, ones, zeros, zeros, zeros + 2], dim=1)  # (N, 6)

    az, el, scale, trans = torch.split(view_para, [1, 1, 1, 3], dim=-1)
    rot = azel2rot(az, el, homo)
    return scale, trans, rot


def azel2rot(az, el, homo=True):
    """
    :param az: (N, 1, (1)). y-axis
    :param el: x-axis
    :return: rot: (N, 4, 4). rotation: Ry? then Rx? x,y,z
    """
    N = az.size(0)
    az = az.view(N, 1, 1)
    el = el.view(N, 1, 1)
    ones = torch.ones_like(az)
    zeros = torch.zeros_like(az)

    # rot = py_t.euler_angles_to_matrix(torch.cat([az.view(N, 1), el.view(N, 1), zeros.view(N, 1)], dim=1),'YXZ')
    # return rot
    batch_rot_y = torch.cat([
        torch.cat([torch.cos(az), zeros, -torch.sin(az)], dim=2),
        torch.cat([zeros, ones, zeros], dim=2),
        torch.cat([torch.sin(az), zeros, torch.cos(az)], dim=2),
    ], dim=1)

    batch_rot_x = torch.cat([
        torch.cat([ones, zeros, zeros], dim=2),
        torch.cat([zeros, torch.cos(el), torch.sin(el)], dim=2),
        torch.cat([zeros, -torch.sin(el), torch.cos(el)], dim=2),
    ], dim=1)
    rotation_matrix = torch.matmul(batch_rot_y, batch_rot_x)
    if homo:
        rotation_matrix = homo_matrix(rotation_matrix)
    return rotation_matrix



def homo_matrix(rot: torch.Tensor):
    """
    :param rot: (N, 3, 3)
    :return: (N, 4, 4)
    """
    device = rot.device
    N = rot.size(0)
    zeros = torch.zeros([N, 1, 1], device=device)
    rotation_matrix = torch.cat([
        torch.cat([rot, torch.zeros(N, 3, 1, device=device)], dim=2),
        torch.cat([zeros, zeros, zeros, zeros + 1], dim=2)
    ], dim=1)
    return rotation_matrix

def homo_to_3x3(rot):
    return rot[:, :3, :3]

# ## Sample
def sample_view(mode, N, device, return_uni=True, use_scale=0, **kwargs):
    sample_dict = {'cfg': _sample_from_cfg,
                   'side': _sample_multi_modal,
                   }
    if mode.startswith('side'):
        if '-' in mode:
            sigma = float(mode.split('-')[-1]) / 180 * np.pi
        else:
            sigma = np.pi / 8
        view_param = sample_dict['side'](N, device, use_scale, sigma=sigma, **kwargs)
    else:
        view_param = sample_dict[mode](N, device, use_scale, **kwargs)

    if return_uni:
        sample_view_trans = azel2uni(view_param)
    else:
        sample_view_trans = None
    return sample_view_trans, view_param


def _sample_multi_modal(N, device, use_scale=0, **kwarg):
    """
    :param view:
    :param other:
    :return: SO3 in shape of (N, 4, 4)
    """
    sample_view = torch.rand(N, 5).to(device)  # [0, 1] in shape (N, 5) excluding az

    cfg = kwarg.get('cfg')

    view_var = torch.FloatTensor([cfg['ele_high'] - cfg['ele_low'],
                                  cfg['scale_high'] - cfg['scale_low'],
                                  0, 0, 0]).to(device).unsqueeze(0)

    view_low = torch.FloatTensor([cfg['ele_low'], cfg['scale_low'],
                                  0, 0, 2]).to(device).unsqueeze(0)
    if use_scale <= 0:
        view_var[:, -4] = 0
        view_low[:, -4] = 1
    sample_view = sample_view * view_var + view_low

    az_mean = torch.randint(0, 2, [N, 1]).to(device) # 0 / 1
    az_mean = az_mean * 2 - 1  # -1 / 1
    az_mean = az_mean * np.pi / 2  # -np.pi/2, np.pi/2

    az_sigma = kwarg.get('sigma', np.pi / 8)
    # az_sigma = np.pi / 8
    sample_az = torch.randn(N, 1).to(device) * az_sigma + az_mean

    sample_view = torch.cat([sample_az, sample_view], dim=1)
    return sample_view


def _sample_from_cfg(N, device, use_scale=0, **kwargs):
    """
    :param view:
    :return: SO3 in shape of (N, 4, 4)
    """
    sample_view = torch.rand(N, 6).to(device)  # [0, 1] in shape (N, 6)
    cfg = kwargs.get('cfg')
    degree = kwargs.get('degree', False)

    if degree:
        cfg['azi_high'] = np.deg2rad(cfg['azi_high'])
        cfg['azi_low']  = np.deg2rad(cfg['azi_low'] )
        cfg['ele_high'] = np.deg2rad(cfg['ele_high'])
        cfg['ele_low'] = np.deg2rad(cfg['ele_low'])
    view_var = torch.FloatTensor([cfg['azi_high'] - cfg['azi_low'],
                                  cfg['ele_high'] - cfg['ele_low'],
                                  cfg['scale_high'] - cfg['scale_low'],
                                  0, 0, 0]).to(device).unsqueeze(0)
    view_low = torch.FloatTensor([cfg['azi_low'], cfg['ele_low'], cfg['scale_low'],
                                  0, 0, 2]).to(device).unsqueeze(0)
    if use_scale <= 0:
        view_var[:, 2] = 0
        view_low[:, 2] = 1
    sample_view = sample_view * view_var + view_low
    return sample_view


def calc_rho(f):
    base_f = 1.875
    base_rho = 2
    rho = base_rho * f / base_f
    return rho


def expand_uni(param, num):
    scale, trans, rot = param
    T = scale.size(0)

    param_list = []
    for t in range(T):
        s = scale[t:t+1].expand(num, 1)
        d = trans[t:t+1].expand(num, 3)
        r = rot[t:t+1].expand(num, 4, 4)
        param_list.append([s, d, r])
    return param_list


def rt_to_homo(rot, t=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :return: (N, 4, 4) [R, t; 0, 1]
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] = 1
    mat = torch.cat([mat, zeros], dim=-2)
    return mat


def diag_to_homo(diag):
    """
    :param diag: (N, )
    :return:
    """
    N = diag.size(0)
    diag = diag.view(N, 1, 1)

    zeros = torch.zeros_like(diag)
    ones = torch.ones_like(diag)
    mat = torch.cat([
        torch.cat([diag, zeros, zeros, zeros], dim=2),
        torch.cat([zeros, diag, zeros, zeros], dim=2),
        torch.cat([zeros, zeros, diag, zeros], dim=2),
        torch.cat([zeros, zeros, zeros, ones], dim=2),
    ], dim=1)
    return mat

