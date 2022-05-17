import torch
import torch.nn as nn
import absl.flags as flags
from absl import app

FLAGS = flags.FLAGS  # can control the weight of each term here

class consistency_loss(nn.Module):
    def __init__(self, beta):
        super(consistency_loss, self).__init__()
        self.loss_func = nn.SmoothL1Loss(beta=beta)

    def forward(self, name_list, pred_list, gt_list):
        loss_list = {}
        if 'nocs_dist_consistency' in name_list:
            loss_list['nocs_dist'] = self.loss_func(pred_list['face_dis_prior'], pred_list['face_dis_pred'])
        return loss_list

    def cal_obj_mask(self, p_mask, g_mask):
        return self.loss_func(p_mask, g_mask.long().squeeze())

