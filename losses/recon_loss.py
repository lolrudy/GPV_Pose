import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
from tools.plane_utils import get_plane, get_plane_parameter
from tools.rot_utils import get_vertical_rot_vec
FLAGS = flags.FLAGS  # can control the weight of each term here

class recon_6face_loss(nn.Module):
    def __init__(self):
        super(recon_6face_loss, self).__init__()
        self.loss_func = nn.L1Loss()

    def forward(self, name_list, pred_list, gt_list, sym, obj_ids, save_path=None):
        loss_list = {}

        if 'Per_point' in name_list:
            res_normal, res_dis, res_f = self.cal_recon_loss_point(gt_list['Points'],
                                                                   pred_list['F_n'],
                                                                   pred_list['F_d'],
                                                                   pred_list['F_c'],
                                                                   gt_list['R'],
                                                                   gt_list['T'],
                                                                   gt_list['Size'],
                                                                   gt_list['Mean_shape'],
                                                                   sym, obj_ids)
            loss_list['recon_per_p'] = FLAGS.recon_n_w * res_normal + FLAGS.recon_d_w * res_dis
            loss_list['recon_p_f'] = FLAGS.recon_f_w * res_f
        if 'Point_voting' in name_list:
            F_c_detach = pred_list['F_c'].detach()
            recon_point_vote, recon_point_r, recon_point_t, recon_point_s, recon_point_self = self.cal_recon_loss_vote(
                                                                             gt_list['Points'],
                                                                             pred_list['F_n'],
                                                                             pred_list['F_d'],
                                                                             F_c_detach,
                                                                             pred_list['Rot1'],
                                                                             pred_list['Rot1_f'],
                                                                             pred_list['Rot2'],
                                                                             pred_list['Rot2_f'],
                                                                             pred_list['Tran'],
                                                                             pred_list['Size'],
                                                                             gt_list['R'],
                                                                             gt_list['T'],
                                                                             gt_list['Size'],
                                                                             gt_list['Mean_shape'],
                                                                             sym, obj_ids, save_path)
            loss_list['recon_point_vote'] = FLAGS.recon_v_w * recon_point_vote
            loss_list['recon_point_r'] = FLAGS.recon_bb_r_w * recon_point_r
            loss_list['recon_point_t'] = FLAGS.recon_bb_t_w * recon_point_t
            loss_list['recon_point_s'] = FLAGS.recon_bb_s_w * recon_point_s
            loss_list['recon_point_self'] = FLAGS.recon_bb_self_w * recon_point_self
        if 'Point_sampling' in name_list:
            loss_list['recon_point_sample'] = FLAGS.recon_s_w * self.cal_recon_loss_sample(pred_list['Pc_sk'],
                                                                                           pred_list['F_c'])

        if 'Point_c_reg' in name_list:
            loss_list['recon_point_c_reg'] = FLAGS.recon_c_w * self.cal_recon_loss_direct(pred_list['F_c'])
        return loss_list

    def cal_recon_loss_direct(self, face_n, face_d, face_c):
        return 0.0


    def cal_recon_loss_sample(self, pc_sk, face_c):
        # I wish that the backbone can directly predict confidence map
        # relative loss ?
        loss_fun = nn.L1Loss()
        res = loss_fun(pc_sk, face_c)
        return res

    def cal_recon_loss_vote(self, pc, face_normal, face_dis, face_c, p_rot_g, f_rot_g, p_rot_r, f_rot_r, p_t, p_s,
                            gt_R, gt_t, gt_s, mean_shape, sym, obj_ids, save_path=None):
        res_vote = 0.0
        res_recon_geo_r = 0.0
        res_recon_geo_t = 0.0
        res_recon_geo_s = 0.0
        res_recon_self_cal = 0.0
        bs = pc.shape[0]
        re_s = gt_s + mean_shape
        pre_s = p_s + mean_shape
        for i in range(bs):
            pc_now = pc[i, ...]
            f_n_now = face_normal[i, ...]  # n x 6 x 3
            f_d_now = face_dis[i, ...]  # n x 6
            f_c_now = face_c[i, ...]  # n x 6
            re_s_now = re_s[i, ...]  # 3
            gt_r_x = gt_R[i, :, 0]
            gt_r_y = gt_R[i, :, 1]
            gt_r_z = gt_R[i, :, 2]
            gt_t_now = gt_t[i, ...]
            obj_id = int(obj_ids[i])
            # y +
            pc_on_plane = pc_now + f_n_now[:, 0, :] * f_d_now[:, 0].view(-1, 1)

            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = fig.add_subplot(121, projection='3d')
            # ax.scatter(ref_points[:, 0], -ref_points[:, 1], ref_points[:, 2], marker='.')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            #
            # ax = fig.add_subplot(122, projection='3d')
            # ax.scatter(view_points[:, 0], -view_points[:, 1], view_points[:, 2], marker='.')
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # plt.show()
            # plt.close()

            # note that in dn_y_up, d also has direction
            n_y_up, dn_y_up, c_y_up = get_plane(pc_on_plane, f_c_now[:, 0])
            if save_path is not None:
                import mmcv, os
                view_points = pc_on_plane.detach().cpu().numpy()
                ref_points = pc_now.detach().cpu().numpy()
                conf_points = f_c_now[:, 0].detach().cpu().numpy()
                import numpy as np
                np.savetxt(save_path + f'_{i}_pc_on_plane_yp.txt', view_points)
                np.savetxt(save_path + f'_{i}_pc_origin_yp.txt', ref_points)
                np.savetxt(save_path + f'_{i}_pc_conf_yp.txt', conf_points)

                plane_parameter = get_plane_parameter(pc_on_plane, f_c_now[:, 0])
                plane_parameter = plane_parameter.detach().cpu().numpy()
                np.savetxt(save_path + f'_{i}_plane_parameter_yp.txt', plane_parameter)

            # cal gt
            dn_gt = gt_r_y * (-(torch.dot(gt_r_y, gt_t_now + gt_r_y * re_s_now[1] / 2)))
            # adjust the sign of n_y_up
            if torch.dot(n_y_up, gt_r_y) < 0:
                n_y_up = -n_y_up
                c_y_up = -c_y_up
            res_yplus = torch.mean(torch.abs(dn_y_up - dn_gt))
            # cal recon_ geo loss

            if sym[i, 0] == 0:
                # x +
                pc_on_plane = pc_now + f_n_now[:, 1, :] * f_d_now[:, 1].view(-1, 1)
                n_x_up, dn_x_up, c_x_up = get_plane(pc_on_plane, f_c_now[:, 1])
                # cal gt
                dn_gt = gt_r_x * (-(torch.dot(gt_r_x, gt_t_now + gt_r_x * re_s_now[0] / 2)))
                # adjust the sign of dn_gt
                if torch.dot(n_x_up, gt_r_x) < 0:
                    n_x_up = -n_x_up
                    c_x_up = -c_x_up
                res_xplus = torch.mean(torch.abs(dn_x_up - dn_gt))
                if save_path is not None:
                    import mmcv, os
                    view_points = pc_on_plane.detach().cpu().numpy()
                    ref_points = pc_now.detach().cpu().numpy()
                    conf_points = f_c_now[:, 1].detach().cpu().numpy()
                    import numpy as np
                    np.savetxt(save_path + f'_{i}_pc_on_plane_xp.txt', view_points)
                    np.savetxt(save_path + f'_{i}_pc_origin_xp.txt', ref_points)
                    np.savetxt(save_path + f'_{i}_pc_conf_xp.txt', conf_points)

                    plane_parameter = get_plane_parameter(pc_on_plane, f_c_now[:, 1])
                    plane_parameter = plane_parameter.detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_plane_parameter_xp.txt', plane_parameter)
                # z +
                pc_on_plane = pc_now + f_n_now[:, 2, :] * f_d_now[:, 2].view(-1, 1)
                n_z_up, dn_z_up, c_z_up = get_plane(pc_on_plane, f_c_now[:, 2])
                # cal gt
                dn_gt = gt_r_z * (-(torch.dot(gt_r_z, gt_t_now + gt_r_z * re_s_now[2] / 2)))
                # adjust the sign of dn_gt
                if torch.dot(n_z_up, gt_r_z) < 0:
                    n_z_up = -n_z_up
                    c_z_up = -c_z_up
                res_zplus = torch.mean(torch.abs(dn_z_up - dn_gt))
                if save_path is not None:
                    import mmcv, os
                    view_points = pc_on_plane.detach().cpu().numpy()
                    ref_points = pc_now.detach().cpu().numpy()
                    conf_points = f_c_now[:, 2].detach().cpu().numpy()
                    import numpy as np
                    np.savetxt(save_path + f'_{i}_pc_on_plane_zp.txt', view_points)
                    np.savetxt(save_path + f'_{i}_pc_origin_zp.txt', ref_points)
                    np.savetxt(save_path + f'_{i}_pc_conf_zp.txt', conf_points)

                    plane_parameter = get_plane_parameter(pc_on_plane, f_c_now[:, 2])
                    plane_parameter = plane_parameter.detach().cpu().numpy()
                    np.savetxt(save_path + f'_{i}_plane_parameter_zp.txt', plane_parameter)

                # x -
                pc_on_plane = pc_now + f_n_now[:, 3, :] * f_d_now[:, 3].view(-1, 1)
                n_x_down, dn_x_down, c_x_down = get_plane(pc_on_plane, f_c_now[:, 3])
                # cal gt
                dn_gt = -gt_r_x * (-(torch.dot(-gt_r_x, gt_t_now - gt_r_x * re_s_now[0] / 2)))
                # adjust the sign of dn_gt
                if torch.dot(n_x_down, -gt_r_x) < 0:
                    n_x_down = -n_x_down
                    c_x_down = -c_x_down
                res_xminus = torch.mean(torch.abs(dn_x_down - dn_gt))
                # z -
                pc_on_plane = pc_now + f_n_now[:, 4, :] * f_d_now[:, 4].view(-1, 1)
                n_z_down, dn_z_down, c_z_down = get_plane(pc_on_plane, f_c_now[:, 4])
                # cal gt
                dn_gt = -gt_r_z * (-(torch.dot(-gt_r_z, gt_t_now - gt_r_z * re_s_now[2] / 2)))
                # adjust the sign of dn_gt
                if torch.dot(n_z_down, -gt_r_z) < 0:
                    n_z_down = -n_z_down
                    c_z_down = -c_z_down
                res_zminus = torch.mean(torch.abs(dn_z_down - dn_gt))
            else:
                res_xplus = 0.0
                res_xminus = 0.0
                res_zplus = 0.0
                res_zminus = 0.0
            # y -
            pc_on_plane = pc_now + f_n_now[:, 5, :] * f_d_now[:, 5].view(-1, 1)
            n_y_down, dn_y_down, c_y_down = get_plane(pc_on_plane, f_c_now[:, 5])
            # cal gt
            dn_gt = -gt_r_y * (-(torch.dot(-gt_r_y, gt_t_now - gt_r_y * re_s_now[1] / 2)))
            # adjust the sign of dn_gt
            if torch.dot(n_y_down, -gt_r_y) < 0:
                n_y_down = -n_y_down
                c_y_down = -c_y_down
            res_yminus = torch.mean(torch.abs(dn_y_down - dn_gt))

            if obj_id != 5:
                res_vote += res_xplus
                res_vote += res_xminus
            res_vote += res_yplus
            res_vote += res_zplus
            res_vote += res_yminus
            res_vote += res_zminus

            #######################cal_ geo recon loss ##################
            # for r, rectify
            new_y, new_x = get_vertical_rot_vec(f_rot_g[i], f_rot_r[i], p_rot_g[i, ...], p_rot_r[i, ...])
            new_z = torch.cross(new_x, new_y)
            # y+
            res_recon_geo_r += torch.mean(torch.abs((n_y_up - new_y)))
            if sym[i, 0] == 0:
                if obj_id != 5:
                    # x+
                    res_recon_geo_r += torch.mean(torch.abs((n_x_up - new_x)))
                    # x-
                    res_recon_geo_r += torch.mean(torch.abs((n_x_down - (-new_x))))
                # z+
                res_recon_geo_r += torch.mean(torch.abs((n_z_up - new_z)))
                # z-
                res_recon_geo_r += torch.mean(torch.abs((n_z_down - (-new_z))))
            # y-
            res_recon_geo_r += torch.mean(torch.abs((n_y_down - (-new_y))))

            # for T
            # Translation must correspond to the center of the bbox
            p_t_now = p_t[i, ...].view(-1)  # 3
            # cal the distance between p_t_now and the predicted plane
            # y+
            dis_y_up = torch.abs(torch.dot(n_y_up, p_t_now) + c_y_up)
            if sym[i, 0] == 0:
                if obj_id != 5:
                    # x+
                    dis_x_up = torch.abs(torch.dot(n_x_up, p_t_now) + c_x_up)
                    # x-
                    dis_x_down = torch.abs(torch.dot(n_x_down, p_t_now) + c_x_down)
                    res_recon_geo_t += torch.abs(dis_x_down - dis_x_up)
                # z+
                dis_z_up = torch.abs(torch.dot(n_z_up, p_t_now) + c_z_up)
                # z-
                dis_z_down = torch.abs(torch.dot(n_z_down, p_t_now) + c_z_down)
                res_recon_geo_t += torch.abs(dis_z_down - dis_z_up)
            # y-
            dis_y_down = torch.abs(torch.dot(n_y_down, p_t_now) + c_y_down)
            res_recon_geo_t += torch.abs(dis_y_down - dis_y_up)

            # for s
            res_recon_geo_s += torch.abs(pre_s[i, 1] / 2.0 - dis_y_down)
            res_recon_geo_s += torch.abs(pre_s[i, 1] / 2.0 - dis_y_up)
            if sym[i, 0] == 0:
                if obj_id != 5:
                    res_recon_geo_s += torch.abs(pre_s[i, 0] / 2.0 - dis_x_down)
                    res_recon_geo_s += torch.abs(pre_s[i, 0] / 2.0 - dis_x_up)
                res_recon_geo_s += torch.abs(pre_s[i, 2] / 2.0 - dis_z_up)
                res_recon_geo_s += torch.abs(pre_s[i, 2] / 2.0 - dis_z_down)

            # for bounding box self-calibrate
            # parallel
            res_recon_self_cal += torch.mean(torch.abs((n_y_up + n_y_down)))
            if sym[i, 0] == 0:
                if obj_id != 5:
                    res_recon_self_cal += torch.mean(torch.abs((n_x_up + n_x_down)))
                res_recon_self_cal += torch.mean(torch.abs((n_z_up + n_z_down)))
            # vertical
            if sym[i, 0] == 0:
                if obj_id != 5:
                    res_recon_self_cal += torch.abs(torch.dot(n_y_up, n_x_up))
                    res_recon_self_cal += torch.abs(torch.dot(n_y_down, n_x_down))
                res_recon_self_cal += torch.abs(torch.dot(n_y_up, n_z_up))
                res_recon_self_cal += torch.abs(torch.dot(n_y_down, n_z_down))

        res_vote = res_vote / 6 / bs
        res_recon_self_cal = res_recon_self_cal / 6 / bs
        res_recon_geo_s = res_recon_geo_s / 6 / bs
        res_recon_geo_r = res_recon_geo_r / 6 / bs
        res_recon_geo_t = res_recon_geo_t / 6 / bs
        return res_vote, res_recon_geo_r, res_recon_geo_t, res_recon_geo_s, res_recon_self_cal


    def cal_recon_loss_point(self, pc, face_normal, face_dis, face_f, gt_R, gt_t, gt_s, mean_shape, sym, obj_ids):
        '''

        :param pc:
        :param face_normal: bs x n x 6 x 3
        :param face_dis: bs x n x 6
        :param face_f: bs x n x 6
        :param gt_R_green: bs x 3
        :param gt_R_red:
        :param gt_t:
        :param gt_s:
        :param mean_shape:
        :param sym:
        :return:
        '''
        # generate gt
        bs = pc.shape[0]

        # face loss
        res_normal = 0.0
        res_dis = 0.0
        res_f = 0.0
        re_s = gt_s + mean_shape
        pc_proj = torch.bmm(gt_R.permute(0, 2, 1), (pc.permute(0, 2, 1) - gt_t.view(bs, 3, 1))).permute(0, 2, 1)
        # stack gt
        for i in range(bs):
            gt_r_x = gt_R[i, :, 0].view(3)
            gt_r_y = gt_R[i, :, 1].view(3)
            gt_r_z = gt_R[i, :, 2].view(3)
            f_n_now = face_normal[i, ...]  # n x 6 x 3
            # face y +
            f_n_yplus = f_n_now[:, 0, :]   # nn x 3
            res_yplus = torch.mean(1.0 - torch.mm(f_n_yplus, gt_r_y.view(3, 1)))
            obj_id = int(obj_ids[i])
            if sym[i, 0] == 0:
                # face x +
                f_n_xplus = f_n_now[:, 1, :]  # nn x 3
                res_xplus = torch.mean(1.0 - torch.mm(f_n_xplus, gt_r_x.view(3, 1)))
                # face z +
                f_n_zplus = f_n_now[:, 2, :]  # nn x 3
                res_zplus = torch.mean(1.0 - torch.mm(f_n_zplus, gt_r_z.view(3, 1)))
                # face x -
                f_n_xminus = f_n_now[:, 3, :]  # nn x 3
                res_xminus = torch.mean(1.0 - torch.mm(f_n_xminus, -gt_r_x.view(3, 1)))
                # face z -
                f_n_zminus = f_n_now[:, 4, :]  # nn x 3
                res_zminus = torch.mean(1.0 - torch.mm(f_n_zminus, -gt_r_z.view(3, 1)))
            else:
                res_xplus = 0.0
                res_xminus = 0.0
                res_zplus = 0.0
                res_zminus = 0.0
            # face y -
            f_n_yminus = f_n_now[:, 5, :]  # nn x 3
            res_yminus = torch.mean(1.0 - torch.mm(f_n_yminus, -gt_r_y.view(3, 1)))

            res_normal += res_xplus
            res_normal += res_yplus
            res_normal += res_zplus
            res_normal += res_xminus
            res_normal += res_yminus
            res_normal += res_zminus
        # dis loss,
            pc_now = pc_proj[i, ...]   # n x 3
            re_s_now = re_s[i, ...]  # 3
            f_d_now = face_dis[i, ...]  # n x 6
            # face y +
            f_d_yplus = f_d_now[:, 0]  # nn x 1
            f_d_gt_yplus = re_s_now[1] / 2 - pc_now[:, 1]
            res_yplus = torch.mean(torch.abs(f_d_yplus - f_d_gt_yplus))
            if sym[i, 0] == 0:
                # face x +
                f_d_xplus = f_d_now[:, 1]  # nn x 1
                f_d_gt_xplus = re_s_now[0] / 2 - pc_now[:, 0]
                res_xplus = torch.mean(torch.abs(f_d_xplus - f_d_gt_xplus))
                # face z +
                f_d_zplus = f_d_now[:, 2]  # nn x 1
                f_d_gt_zplus = re_s_now[2] / 2 - pc_now[:, 2]
                res_zplus = torch.mean(torch.abs(f_d_zplus - f_d_gt_zplus))
                # face x -
                f_d_xminus = f_d_now[:, 3]  # nn x 1
                f_d_gt_xminus = pc_now[:, 0] + re_s_now[0] / 2
                res_xminus = torch.mean(torch.abs(f_d_xminus - f_d_gt_xminus))
                # face z -
                f_d_zminus = f_d_now[:, 4]  # nn x 1
                f_d_gt_zminus = pc_now[:, 2] + re_s_now[2] / 2
                res_zminus = torch.mean(torch.abs(f_d_zminus - f_d_gt_zminus))
            else:
                res_xplus = 0.0
                res_xminus = 0.0
                res_zplus = 0.0
                res_zminus = 0.0
            # face y -
            f_d_yminus = f_d_now[:, 5]  # nn x 1
            f_d_gt_yminus = pc_now[:, 1] + re_s_now[1] / 2
            res_yminus = torch.mean(torch.abs(f_d_yminus - f_d_gt_yminus))

            if obj_id != 5:
                res_dis += res_xplus
                res_dis += res_xminus
            res_dis += res_yplus
            res_dis += res_zplus
            res_dis += res_yminus
            res_dis += res_zminus

        # face_c loss
            # face y+
            c_y_up = face_f[i, :, 0]
            cc_y_up = torch.norm(f_n_yplus * f_d_yplus.view(-1, 1) -
                                 gt_r_y.view(1, 3).repeat(f_n_yplus.shape[0], 1) * f_d_gt_yplus.view(-1, 1), dim=1)
            f_y_up = torch.exp(-303.5 *cc_y_up * cc_y_up)
            res_f += torch.mean(torch.abs(f_y_up - c_y_up))
            if sym[i, 0] == 0:
                if obj_id != 5:
                    # face x+
                    c_x_up = face_f[i, :, 1]
                    cc_x_up = torch.norm(
                        f_n_xplus * f_d_xplus.view(-1, 1) -
                        gt_r_x.view(1, 3).repeat(f_n_xplus.shape[0], 1) * f_d_gt_xplus.view(-1, 1), dim=1)
                    f_x_up = torch.exp(-303.5 * cc_x_up * cc_x_up)
                    res_f += torch.mean(torch.abs(f_x_up - c_x_up))

                    # face x-
                    c_x_down = face_f[i, :, 3]
                    cc_x_down = torch.norm(
                        f_n_xminus * f_d_xminus.view(-1, 1) -
                        (-gt_r_x).view(1, 3).repeat(f_n_xminus.shape[0], 1) * f_d_gt_xminus.view(-1, 1), dim=1)
                    f_x_down = torch.exp(-303.5 * cc_x_down * cc_x_down)
                    res_f += torch.mean(torch.abs(f_x_down - c_x_down))

                # face z+
                c_z_up = face_f[i, :, 2]
                cc_z_up = torch.norm(
                    f_n_zplus * f_d_zplus.view(-1, 1) -
                    gt_r_z.view(1, 3).repeat(f_n_zplus.shape[0], 1) * f_d_gt_zplus.view(-1, 1), dim=1)
                f_z_up = torch.exp(-303.5 * cc_z_up * cc_z_up)
                res_f += torch.mean(torch.abs(f_z_up - c_z_up))

                # face z-
                c_z_down = face_f[i, :, 4]
                cc_z_down = torch.norm(
                    f_n_zminus * f_d_zminus.view(-1, 1) -
                    (-gt_r_z).view(1, 3).repeat(f_n_zminus.shape[0], 1) * f_d_gt_zminus.view(-1, 1), dim=1)
                f_z_down = torch.exp(-303.5 * cc_z_down * cc_z_down)
                res_f += torch.mean(torch.abs(f_z_down - c_z_down))

            # face y-
            c_y_down = face_f[i, :, 5]
            cc_y_down = torch.norm(
                f_n_yminus * f_d_yminus.view(-1, 1) -
                (-gt_r_y).view(1, 3).repeat(f_n_yminus.shape[0], 1) * f_d_gt_yminus.view(-1, 1), dim=1)
            f_y_down = torch.exp(-303.5 * cc_y_down * cc_y_down)
            res_f += torch.mean(torch.abs(f_y_down - c_y_down))

        res_dis = res_dis / 6 / bs
        res_normal = res_normal / 6 / bs
        res_f = res_f / 6 / bs
        return res_normal, res_dis, res_f

