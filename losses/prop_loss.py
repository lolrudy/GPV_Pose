import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
from tools.rot_utils import get_rot_vec_vert_batch, get_rot_mat_y_first, get_vertical_rot_vec

FLAGS = flags.FLAGS  # can control the weight of each term here


class prop_rot_loss(nn.Module):
    def __init__(self):
        super(prop_rot_loss, self).__init__()
        self.loss_func = nn.L1Loss()

    def forward(self, namelist, pred_list, gt_list, sym):
        loss_list = {}
        if "Prop_pm" in namelist:
            loss_list["Prop_pm"] = FLAGS.prop_pm_w * self.prop_point_matching_loss(gt_list['Points'],
                                                                                   pred_list['Rot1'],
                                                                                   pred_list['Rot1_f'],
                                                                                   pred_list['Rot2'],
                                                                                   pred_list['Rot2_f'],
                                                                                   pred_list['Tran'],
                                                                                   gt_list['R'],
                                                                                   gt_list['T'],
                                                                                   sym)

        if "Prop_r_reg" in namelist:
            loss_list["Prop_r_reg"] = FLAGS.prop_r_reg_w * self.prop_rot_reg_loss(pred_list['Rot1_f'],
                                                                                  pred_list['Rot2_f'])

        if "Prop_sym" in namelist and (FLAGS.prop_sym_w > 0):
             Prop_sym_recon, Prop_sym_rt = self.prop_sym_matching_loss(gt_list['Points'],
                                                                       pred_list['Recon'],
                                                                       pred_list['Rot1'],
                                                                       pred_list['Rot2'],
                                                                       pred_list['Tran'],
                                                                       gt_list['R'],
                                                                       gt_list['T'],
                                                                       sym)
             loss_list["Prop_sym_recon"] = FLAGS.prop_sym_w * Prop_sym_recon
             loss_list["Prop_sym_rt"] = FLAGS.prop_sym_w * Prop_sym_rt
        else:
            loss_list["Prop_occ"] = 0.0
        return loss_list

        # for mug, this loss is different
    def prop_sym_matching_loss(self, PC, PC_re, p_g_vec, p_r_vec, p_t, gt_R, gt_t, sym):
        bs = PC.shape[0]
        points_re_cano = torch.bmm(gt_R.permute(0, 2, 1), (PC - gt_t.view(bs, 1, -1)).permute(0, 2, 1))
        points_re_cano = points_re_cano.permute(0, 2, 1)
        res_p_recon = 0.0
        res_p_rt = 0.0
        for i in range(bs):
            PC_now = PC[i, ...]
            PC_re_now = PC_re[i, ...]
            PC_re_cano = points_re_cano[i, ...]
            p_g_now = p_g_vec[i, ...]
            if sym[i, 0] == 1 and torch.sum(sym[i, 1:]) > 0:  # y axis reflection, can, bowl, bottle
                gt_re_points = torch.cat([-PC_re_cano[:, 0].view(-1, 1),
                                          PC_re_cano[:, 1].view(-1, 1),
                                          -PC_re_cano[:, 2].view(-1, 1)], dim=1)
                gt_PC = torch.mm(gt_R[i, ...], gt_re_points.T) + gt_t[i, ...].view(-1, 1)
                gt_PC = gt_PC.T    # 1028 X 3
                res_p_recon += self.loss_func(PC_re_now, gt_PC)
                # loss for p_g_now
                #p_g_now= gt_R[i, :, 1]
                p_t_now = p_t[i, ...]
                pc_t_res = PC_now - p_t_now.view(1, -1)
                vec_along_p_g = torch.mm(torch.mm(pc_t_res, p_g_now.view(-1, 1)), p_g_now.view(1, -1))  # 1028 x 3
                a_to_1_2_b = vec_along_p_g - pc_t_res
                PC_b = PC_now + 2.0 * a_to_1_2_b
                res_p_rt += self.loss_func(PC_b, PC_re_now)
            elif sym[i, 0] == 0 and sym[i, 1] == 1:  # yx reflection, laptop, mug with handle
                gt_re_points = torch.cat([PC_re_cano[:, 0].view(-1, 1),
                                          PC_re_cano[:, 1].view(-1, 1),
                                          -PC_re_cano[:, 2].view(-1, 1)], dim=1)
                gt_PC = torch.mm(gt_R[i, ...], gt_re_points.T) + gt_t[i, ...].view(-1, 1)
                gt_PC = gt_PC.T  # 1028 X 3
                res_p_recon += self.loss_func(PC_re_now, gt_PC)
                # loss for p_g_now and p_r_now
                p_r_now = p_r_vec[i, ...]
                #p_r_now = gt_R[i, :, 0]
                #p_g_now = gt_R[i, :, 1]
                p_t_now = p_t[i, ...]
                # cal plane normal
                p_z_now = torch.cross(p_r_now, p_g_now)
                p_z_now = p_z_now / (torch.norm(p_z_now) + 1e-8)
                #  please refer to https://www.zhihu.com/question/450901026
                t = -(torch.mm(p_z_now.view(1, -1), PC_now.T) - torch.dot(p_z_now, p_t_now))  # 1 x  1028
                PC_b = PC_now + 2.0 * torch.mm(p_z_now.view(-1, 1), t).T
                res_p_rt += self.loss_func(PC_b, PC_re_now)
            elif sym[i, 0] == 1 and torch.sum(sym[i, 1:]) == 0:   # mug without handle
                continue      # not calculate it
            else:       # camera
                res_p_recon += self.loss_func(PC, PC_re)
                res_p_rt += 0.0

        res_p_rt = res_p_rt / bs
        res_p_recon = res_p_recon / bs
        return res_p_recon, res_p_rt


    def prop_rot_reg_loss(self, f_g_vec, f_r_vec):
        res = torch.mean(torch.abs((1.0 - (f_g_vec + f_r_vec))))
        return res

    def prop_point_matching_loss(self, points, p_g_vec, f_g_vec, p_r_vec, f_r_vec, p_t, g_R, g_t, sym):
        # Notice that this loss function do not backpropagate the grad of f_g_vec and f_r_vec
        bs = points.shape[0]
        # reproject the points back to objct coordinate
        # only for non-symmetric objects
        # handle c1, c2
        res = 0.0
        points_re = torch.bmm(g_R.permute(0, 2, 1), (points - g_t.view(bs, 1, -1)).permute(0, 2, 1))
        points_re = points_re.permute(0, 2, 1)
        for i in range(bs):
            if sym[i, 0] == 1:
                # estimate pred_R
                new_y, new_x = get_vertical_rot_vec(f_g_vec[i], 1e-5, p_g_vec[i, ...], g_R[i, :, 0])
                p_R = get_rot_mat_y_first(new_y.view(1, -1), new_x.view(1, -1))[0]  # 3 x 3
                points_re_n = torch.mm(p_R.T, (points[i, ...] - p_t[i, ...].view(1, -1)).T).T  # 1024, 3
                res += self.loss_func(points_re_n, points_re[i, ...])
            else:
                # estimate pred_R
                new_y, new_x = get_vertical_rot_vec(f_g_vec[i], f_r_vec[i], p_g_vec[i, ...], p_r_vec[i, ...])
                p_R = get_rot_mat_y_first(new_y.view(1, -1), new_x.view(1, -1))[0]  # 3 x 3
                points_re_n = torch.mm(p_R.T, (points[i, ...] - p_t[i, ...].view(1, -1)).T).T  # 1024, 3
                res += self.loss_func(points_re_n, points_re[i, ...])
        res = res / bs
        return res






