import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import absl.flags as flags
from absl import app
import tools.geom_utils as g
import tools.image_utils as i_util

FLAGS = flags.FLAGS

class Perspective3d(nn.Module):
    def __init__(self, camK, z_range=1, R=None, t=None, s=None, det=None):
        super(Perspective3d, self).__init__()
        self.z_range = 1
        self.fx = camK[0, 0]
        self.fy = camK[1, 1]
        self.ux = camK[0, 2]
        self.uy = camK[1, 2]
        self.rot = R   # from camera center to current object position
        self.tran = t
        self.s = s     # scale
        self.det = det   #  compensation of the segmentation   bs, 2

    # in this function, the scale is set to 1 by default
    def ray2grid(self, xy_sample, z_sample, bs, device):
        height = width = xy_sample

        x_t = torch.linspace(0, width, width, dtype=torch.float32, device=device)  # image space
        y_t = torch.linspace(0, height, height, dtype=torch.float32, device=device)  # image space
        z_t = torch.linspace(-self.z_range / 2, self.z_range / 2, z_sample, dtype=torch.float32,
                             device=device)  # depth step

        z_t, y_t, x_t = torch.meshgrid(z_t, y_t, x_t)  # [D, W, H]  # cmt: this must be in ZYX order

        x_t = x_t.unsqueeze(0).repeat(bs, 1, 1, 1)
        y_t = y_t.unsqueeze(0).repeat(bs, 1, 1, 1)
        z_t = z_t.unsqueeze(0).repeat(bs, 1, 1, 1)

        Z_t = z_t + self.tran[..., -1]

        X_t = (x_t - self.ux + self.det[..., 0]) * Z_t / self.fx
        Y_t = (y_t - self.uy + self.det[..., 1]) * Z_t / self.fy

        ones = torch.ones_like(X_t)
        grid = torch.stack([X_t, Y_t, Z_t, ones], dim=-1)

        return grid

    def camera2world(self):
        rot_T = (g.homo_to_3x3(self.rot)).permute(0, 2, 1)
        rt_inv = g.rt_to_homo(rot_T, -torch.matmul(rot_T, self.tran.unsqueeze(-1)))
        scale_inv = g.diag_to_homo(1 / self.s)
        wTc = torch.matmul(scale_inv, rt_inv)
        return wTc


    def forward(self, voxels, xy_sample, z_sample):
        bs = voxels.shape[0]
        wTc = self.camera2world()
        cGrid = self.ray2grid(xy_sample, z_sample, bs, device=voxels.device)
        wGrid = torch.matmul(cGrid.view(bs, -1, 4), wTc.transpose(1, 2)).view(bs, z_sample, xy_sample, xy_sample, 4)
        wGrid = 2 * wGrid[..., 0:3] / wGrid[..., 3:4]  # scale from [0.5, 0.5] to [-1, 1]
        voxels = F.grid_sample(voxels, wGrid, align_corners=True)
        return voxels


def main(_):
    device = 'cuda'

    H = W = D = 16
    N = 1
    vox = torch.zeros([N, 1, D, H, W], device=device)
    vox[..., 0:D // 2, 0:H//2, 0:W//2] = 1

    for i in range(-30, 30, 10):
        param = torch.FloatTensor([[0, i / 180 * 3.14, 1, 0, 0, 2]]).to(device)
        scale, tran, rot = g.azel2uni(param)
        f = 375
        camK = torch.tensor([[f, 0.0, 128], [0.0, f, 128], [0, 0, 1]], dtype=torch.float32).to(device)
        det = torch.tensor([[100, 100]], dtype=torch.float32).to(device)
        det = det.view(1, 2)
        IH = 32
        layer = Perspective3d(camK=camK, z_range=1, R=rot, t=tran, s=scale, det=det).to(device)
        trans_vox = layer(vox, IH, IH)
        mask = torch.mean(trans_vox, dim=2)  # (N, 1, H, W)

        save_dir = 'outputs'
        i_util.save_images(mask, os.path.join(save_dir, 'test_%d' % i))

if __name__ == "__main__":
    app.run(main)




