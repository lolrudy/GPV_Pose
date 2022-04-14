import torch
import torch.nn as nn
import torch.optim as optim
import absl.flags as flags
from absl import app
import numpy as np
import torch.nn.functional as F

from network.fs_net_repo.PoseR import Rot_red, Rot_green
from network.fs_net_repo.PoseTs import Pose_Ts
from network.fs_net_repo.FaceRecon import FaceRecon

FLAGS = flags.FLAGS

class PoseNet9D(nn.Module):
    def __init__(self):
        super(PoseNet9D, self).__init__()
        self.rot_green = Rot_green()
        self.rot_red = Rot_red()
        self.face_recon = FaceRecon()
        self.ts = Pose_Ts()

    def forward(self, points, obj_id):
        bs, p_num = points.shape[0], points.shape[1]
        recon, face, feat = self.face_recon(points - points.mean(dim=1, keepdim=True), obj_id)
        recon = recon + points.mean(dim=1, keepdim=True)
        # handle face
        face_normal = face[:, :, :18].view(bs, p_num, 6, 3)  # normal
        face_normal = face_normal / torch.norm(face_normal, dim=-1, keepdim=True)  # bs x nunm x 6 x 3
        face_dis = face[:, :, 18:24]  # bs x num x  6
        face_f = F.sigmoid(face[:, :, 24:])  # bs x num x 6
        #  rotation
        green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 4
        red_R_vec = self.rot_red(feat.permute(0, 2, 1))   # b x 4
        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])

        # translation and size
        feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
        T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        Pred_T = T + points.mean(dim=1)  # bs x 3
        Pred_s = s  # this s is not the object size, it is the residual

        return recon, face_normal, face_dis, face_f, p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s


def main(argv):
    classifier_seg3D = PoseNet9D()

    points = torch.rand(2, 1000, 3)
    import numpy as np
    obj_idh = torch.ones((2, 1))
    obj_idh[1, 0] = 5
    '''
    if obj_idh.shape[0] == 1:
        obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)
    else:
        obj_idh = obj_idh.view(-1, 1)

    one_hot = torch.zeros(points.shape[0], 6).scatter_(1, obj_idh.cpu().long(), 1)
    '''
    recon, f_n, f_d, f_f, r1, r2, c1, c2, t, s = classifier_seg3D(points, obj_idh)
    t = 1



if __name__ == "__main__":
    print(1)
    from config.config import *
    app.run(main)





