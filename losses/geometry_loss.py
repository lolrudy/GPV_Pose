import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
from tools.rot_utils import get_vertical_rot_vec, get_rot_mat_y_first

FLAGS = flags.FLAGS  # can control the weight of each term here


class geo_transform_loss(nn.Module):
    def __init__(self):
        super(geo_transform_loss, self).__init__()
        self.loss_func = nn.L1Loss()

    def forward(self, name_list, pred_list, gt_list, sym):
        loss_list = {}

        if 'Geo_point' in name_list:
            loss_list['geo_point'] = FLAGS.geo_p_w * self.cal_geo_loss_point(gt_list['Points'],
                                                                             pred_list['Rot1'],
                                                                             pred_list['Rot2'],
                                                                             pred_list['Tran'],
                                                                             gt_list['R'],
                                                                             gt_list['T'], sym)

        if 'Geo_face' in name_list:
            loss_list['geo_face'] = FLAGS.geo_f_w * self.cal_geo_loss_face(gt_list['Points'],
                                                                           pred_list['Rot1'],
                                                                           pred_list['Rot1_f'],
                                                                           pred_list['Rot2'],
                                                                           pred_list['Rot2_f'],
                                                                           pred_list['Tran'],
                                                                           pred_list['Size'],
                                                                           gt_list['Mean_shape'], sym)
        return loss_list

    def cal_geo_loss_face(self, points, p_rot_g, f_rot_g, p_rot_r, f_rot_r, p_t, p_s, mean_shape, sym):
        bs = points.shape[0]
        res = 0.0
        re_s = p_s + mean_shape
        for i in range(bs):
            # reproj
            # cal R
            new_y, new_x = get_vertical_rot_vec(f_rot_g[i], f_rot_r[i], p_rot_g[i, ...], p_rot_r[i, ...])
            p_R = get_rot_mat_y_first(new_y.view(1, -1), new_x.view(1, -1))[0]  # 3 x 3
            points_re = torch.mm(p_R.permute(1, 0), (points[i, ...] - p_t[i, ...].view(1, -1)).permute(1, 0))
            points_re = points_re.permute(1, 0)  # n x 3
            # points_reaugment according to the sym
            if sym[i, 1] > 0:  # xy reflection
                points_re_z = torch.cat([points_re[:, :2], -points_re[:, 2:]], dim=-1)
                points_re = torch.cat([points_re, points_re_z], dim=0)
            if sym[i, 2] > 0:  # xz reflection
                points_re_z = torch.cat([points_re[:, 0].view(-1, 1), -points_re[:, 1].view(-1, 1),
                                         points_re[:, 2].view(-1, 1)], dim=-1)
                points_re = torch.cat([points_re, points_re_z], dim=0)
            if sym[i, 3] > 0:  # yz reflection
                points_re_z = torch.cat([-points_re[:, 0].view(-1, 1), points_re[:, 1:]], dim=-1)
                points_re = torch.cat([points_re, points_re_z], dim=0)

            # for six faces
            # face 1
            residuals = torch.abs(re_s[i, 1] / 2 - points_re[:, 1])
            res_yplus = torch.min(residuals)
            # face 2
            residuals = torch.abs(re_s[i, 0] / 2 - points_re[:, 0])
            res_xplus = torch.min(residuals)
            # face 3
            residuals = torch.abs(re_s[i, 2] / 2 - points_re[:, 2])
            res_zplus = torch.min(residuals)
            # face 4
            residuals = torch.abs(points_re[:, 0] + re_s[i, 0] / 2)
            res_xminus = torch.min(residuals)
            # face 5
            residuals = torch.abs(points_re[:, 2] + re_s[i, 2] / 2)
            res_zminus = torch.min(residuals)
            # face 6
            residuals = torch.abs(points_re[:, 1] + re_s[i, 1] / 2)
            res_yminus = torch.min(residuals)

            res += res_xplus
            res += res_yplus
            res += res_zplus
            res += res_xminus
            res += res_yminus
            res += res_zminus
        res = res / 6 / bs
        return res

    def cal_geo_loss_point(self, points, p_rot_g, p_rot_r, p_t, g_R, g_t, sym):
        '''

        :param points: the selected sketch points
        :param p_rot_g: green vec
        :param p_rot_r: red_vec_direction
        :return:
        '''

        bs = points.shape[0]
        # reproject the points back to objct coordinate
        points_re = torch.bmm(g_R.permute(0, 2, 1), (points - g_t.view(bs, 1, -1)).permute(0, 2, 1))
        points_re = points_re.permute(0, 2, 1)
        # reproject by p_rot_green
        points_re_y = torch.sum((points - p_t.view(bs, 1, -1)) * p_rot_g.view(bs, 1, -1), dim=-1)
        res_geo_y = self.loss_func(points_re_y, points_re[:, :, 1])
        # depends on symmetry
        res_geo_x = 0.0
        valid = 0
        for i in range(bs):
            if sym[i, 0] == 1:
                continue
            else:
                points_re_x = torch.sum((points[i, ...] - p_t[i, ...].view(1, -1)) * p_rot_r[i, ...].view(1, -1), dim=-1)
                tmp_res_x = self.loss_func(points_re_x, points_re[i, :, 0])
                res_geo_x += torch.mean(tmp_res_x)
                valid += 1
        if valid > 0:
            res_geo_x = res_geo_x / valid
        else:
            res_geo_x = 0.0
        return res_geo_y + res_geo_x
