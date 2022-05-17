import torch
import torch.nn as nn
import absl.flags as flags

FLAGS = flags.FLAGS

from network.fs_net_repo.PoseNet9D import PoseNet9D
from network.point_sample.pc_sample import PC_sample, PC_sample_v2
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc
from losses.fs_net_loss import fs_net_loss
from losses.recon_loss import recon_6face_loss
from losses.geometry_loss import geo_transform_loss
from losses.prop_loss import prop_rot_loss
from losses.shape_prior_loss import shape_prior_loss
from losses.consistency_loss import consistency_loss
from engine.organize_loss import control_loss
from tools.training_utils import get_gt_v
from tools.shape_prior_utils import get_point_depth_error, get_nocs_model
import os


class GPVPose(nn.Module):
    def __init__(self, train_stage):
        super(GPVPose, self).__init__()
        self.pose_net = PoseNet9D()
        self.train_stage = train_stage
        self.loss_recon = recon_6face_loss()
        self.loss_fs_net = fs_net_loss()
        self.loss_geo = geo_transform_loss()
        self.loss_prop = prop_rot_loss()
        if FLAGS.use_shape_prior_loss:
            self.loss_shape_prior = shape_prior_loss(FLAGS.prior_corr_wt, FLAGS.prior_cd_wt, FLAGS.prior_entropy_wt, FLAGS.prior_deform_wt, FLAGS.prior_corr_threshold,
                                                     FLAGS.prior_sym_wt)
        else:
            self.loss_shape_prior = shape_prior_loss(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.loss_consistency = consistency_loss(beta=FLAGS.consistency_beta)
        self.name_fs_list, self.name_recon_list, self.name_geo_list, \
            self.name_prop_list = control_loss(self.train_stage)

    def forward(self, depth, obj_id, camK, mean_shape,
                gt_R=None, gt_t=None, gt_s_delta=None, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None, do_loss=False, rgb=None,
                depth_normalize=None, gt_mask=None, shape_prior=None, nocs_coord=None, logger=None, batch_num=None):

        output_dict = {}

        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        seman_feature = torch.zeros((bs,H,W,32)).float().to(rgb.device)
        obj_mask_output = obj_mask = None

        if self.train_stage in ['PoseNet_only', 'shape_prior_only', 'FSNet_only']:
            FLAGS.sample_method = 'basic'

            sketch = torch.rand([bs, 6, H, W], device=depth.device)

            PC, PC_sk, PC_seman, PC_nocs = PC_sample_v2(sketch, def_mask, depth, camK, gt_2D, seman_feature, nocs_coord)
            if FLAGS.train:
                gt_s = gt_s_delta + mean_shape
                # save_dir = 'output/check_augment_result_2mm_origin/'
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                # i = 0
                # while os.path.exists(save_dir + f'{i}_0_pc.txt'):
                #     i = i + 1
                # save_path = os.path.join(save_dir, f'{i}')
                point_depth_error, point_nocs_error = get_point_depth_error(PC_nocs, PC, gt_R, gt_t, gt_s, model=model_point, save_path=None)
                point_mask_gt = point_depth_error < FLAGS.point_mask_distance_threshold
                point_mask_gt = point_mask_gt.detach()
                is_data_valid = True
                invalid_list = []
                for i in range(bs):
                    # save invalid file
                    p_valid_num = torch.sum(point_mask_gt[i])
                    if p_valid_num < FLAGS.point_mask_min_threshold:
                        output_invalid_image_dir = os.path.join(FLAGS.model_save, 'invalid_image')
                        if not os.path.exists(output_invalid_image_dir):
                            os.makedirs(output_invalid_image_dir)
                        try_num = 0
                        while True:
                            output_rgb_name = f'invalid_num_{p_valid_num}_{try_num}_rgb.npy'
                            if os.path.exists(os.path.join(output_invalid_image_dir, output_rgb_name)):
                                try_num += 1
                            else:
                                break
                        rgb_i = rgb[i].detach().cpu().numpy()
                        depth_i = depth[i].detach().cpu().numpy()
                        coord_i = nocs_coord[i].detach().cpu().numpy()
                        import numpy as np
                        np.save(os.path.join(output_invalid_image_dir, output_rgb_name), rgb_i)
                        np.save(os.path.join(output_invalid_image_dir, output_rgb_name.replace('rgb', 'depth')), depth_i)
                        np.save(os.path.join(output_invalid_image_dir, output_rgb_name.replace('rgb', 'coord')), coord_i)
                        if logger is not None:
                            logger.warning(f'WARNING: valid point num:{p_valid_num}')
                        else:
                            print(f'WARNING: valid point num:{p_valid_num}')
                        is_data_valid = False
                        invalid_list.append(i)
                    output_dict['invalid_list'] = invalid_list
            else:
                is_data_valid = True

            if PC is None or (not is_data_valid):
                return output_dict, None

            if PC.isnan().any():
                if logger is not None:
                    logger.warning('nan detect in point cloud!!')
                else:
                    print('nan detect in point cloud!!')
                return output_dict, None

            PC = PC.detach()
            if FLAGS.train:
                PC_da, gt_R_da, gt_t_da, gt_s_da, model_point, PC_nocs = self.data_augment(PC, gt_R, gt_t, gt_s_delta,
                                                                                           mean_shape, sym,
                                                                                           aug_bb,
                                                                                           aug_rt_t, aug_rt_r,
                                                                                           model_point, nocs_scale,
                                                                                           PC_nocs,
                                                                                           obj_id)
                PC = PC_da
                gt_R = gt_R_da
                gt_t = gt_t_da
                gt_s_delta = gt_s_da
                gt_s = gt_s_delta + mean_shape
            else:
                gt_s = None
            recon, face_shift, face_shift_delta, face_shift_prior, face_log_var, p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s,\
            Pred_s_delta, assign_mat, deform_field, nocs_pred, point_mask_conf \
                = self.pose_net(
                PC, obj_id, shape_prior, PC_seman, mean_shape, gt_s=gt_s)
            if Pred_T.isnan().any() or p_green_R.isnan().any() or p_red_R.isnan().any():
                if logger is not None:
                    logger.warning('nan detect in trans / rot!!')
                else:
                    print('nan detect in trans / rot!!')
                return output_dict, None

            output_dict['mask'] = obj_mask_output
            output_dict['p_green_R'] = p_green_R
            output_dict['p_red_R'] = p_red_R
            output_dict['f_green_R'] = f_green_R
            output_dict['f_red_R'] = f_red_R
            output_dict['Pred_T'] = Pred_T
            output_dict['Pred_s'] = Pred_s
        elif self.train_stage == 'seman_encoder_only':
            output_dict['mask'] = obj_mask_output
            recon, face_normal, face_dis, face_log_var, p_green_R, p_red_R, f_green_R, f_red_R, \
            Pred_T, Pred_s, Pred_s_delta, assign_mat, deform_field, nocs_pred, face_dis_delta, point_mask_conf, \
            PC, PC_sk, PC_seman, PC_nocs, point_mask_gt, gt_s \
                = None, None, None, None, None, None, None, None, \
                    None, None, None, None, None, None, None, None, None, None, None, None, None, None
        else:
            raise NotImplementedError



        if do_loss:
            p_recon = recon
            p_T = Pred_T
            p_s = Pred_s
            pred_fsnet_list = {
                'Rot1': p_green_R,
                'Rot1_f': f_green_R,
                'Rot2': p_red_R,
                'Rot2_f': f_red_R,
                'Recon': p_recon,
                'Tran': p_T,
                'Size': p_s,
            }

            if self.train_stage == 'Backbone_only':
                gt_green_v = None
                gt_red_v = None
            else:
                gt_green_v, gt_red_v = get_gt_v(gt_R)

            gt_fsnet_list = {
                'Rot1': gt_green_v,
                'Rot2': gt_red_v,
                'Recon': PC,
                'Tran': gt_t,
                'Size': gt_s,
            }
            fsnet_loss = self.loss_fs_net(self.name_fs_list, pred_fsnet_list, gt_fsnet_list, sym)

            # prop loss
            pred_prop_list = {
                'Recon': p_recon,
                'Rot1': p_green_R,
                'Rot2': p_red_R,
                'Tran': p_T,
                'Scale': p_s,
                'Rot1_f': f_green_R.detach(),
                'Rot2_f': f_red_R.detach(),
            }

            gt_prop_list = {
                'Points': PC,
                'R': gt_R,
                'T': gt_t,
                'Mean_shape': mean_shape,
            }
            prop_loss = self.loss_prop(self.name_prop_list, pred_prop_list, gt_prop_list, sym)

            recon_loss = self.loss_recon(self.name_recon_list, [], [], sym, obj_id)

            # geo loss
            pred_geo_list = {
                'Rot1': p_green_R,
                'Rot2': p_red_R,
                'Tran': p_T,
                'Size': p_s,
                'Rot1_f': f_green_R.detach(),
                'Rot2_f': f_red_R.detach(),
            }

            gt_geo_list = {
                'Points': PC,
                'R': gt_R,
                'T': gt_t,
                'Mean_shape': mean_shape,
            }
            geo_loss = self.loss_geo(self.name_geo_list, pred_geo_list, gt_geo_list, sym)
            if FLAGS.use_shape_prior_loss:
                if not FLAGS.prior_corr_sym:
                    sym = None
                prior_loss = self.loss_shape_prior(assign_mat, deform_field, shape_prior, PC_nocs, model_point,
                                                   point_mask_gt, sym)
            else:
                prior_loss = {}

            loss_dict = {}
            loss_dict['fsnet_loss'] = fsnet_loss
            loss_dict['recon_loss'] = recon_loss
            loss_dict['geo_loss'] = geo_loss
            loss_dict['prop_loss'] = prop_loss
            loss_dict['prior_loss'] = prior_loss

        else:
            return output_dict

        return output_dict, loss_dict

    def data_augment(self, PC, gt_R, gt_t, gt_s_delta, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                     model_point, nocs_scale, PC_nocs, obj_ids):
        bs = PC.shape[0]
        for i in range(bs):
            obj_id = int(obj_ids[i])
            prop_bb = torch.rand(1)
            if prop_bb < FLAGS.aug_bb_pro:
                #   R, t, s, s_x=(0.9, 1.1), s_y=(0.9, 1.1), s_z=(0.9, 1.1), sym=None
                PC_new, gt_s_new, nocs_new, model_new = defor_3D_bb(PC[i, ...], gt_R[i, ...],
                                                                    gt_t[i, ...],
                                                                    gt_s_delta[i, ...] + mean_shape[i, ...],
                                                                    PC_nocs[i, ...], model_point[i, ...],
                                                                    sym=sym[i, ...], aug_bb=aug_bb[i, ...])
                gt_s_new_delta = gt_s_new - mean_shape[i, ...]
                PC[i, ...] = PC_new
                gt_s_delta[i, ...] = gt_s_new_delta
                PC_nocs[i, ...] = nocs_new
                model_point[i, ...] = model_new

            prop_rt = torch.rand(1)
            if prop_rt < FLAGS.aug_rt_pro:
                PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC[i, ...], gt_R[i, ...],
                                                         gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
                PC[i, ...] = PC_new
                gt_R[i, ...] = gt_R_new
                gt_t[i, ...] = gt_t_new.view(-1)

            prop_bc = torch.rand(1)
            # only do bc for mug and bowl
            if prop_bc < FLAGS.aug_bc_pro and (obj_id == 5 or obj_id == 1):
                PC_new, gt_s_new, model_point_new, nocs_new = defor_3D_bc(PC[i, ...], gt_R[i, ...], gt_t[i, ...],
                                                                          gt_s_delta[i, ...] + mean_shape[i, ...],
                                                                          model_point[i, ...], nocs_scale[i, ...],
                                                                          PC_nocs[i, ...])
                gt_s_new = gt_s_new - mean_shape[i, ...]
                PC[i, ...] = PC_new
                gt_s_delta[i, ...] = gt_s_new
                model_point[i, ...] = model_point_new
                PC_nocs[i, ...] = nocs_new

            prop_pc = torch.rand(1)
            if prop_pc < FLAGS.aug_pc_pro:
                PC_new = defor_3D_pc(PC[i, ...], FLAGS.aug_pc_r)
                PC[i, ...] = PC_new

        return PC, gt_R, gt_t, gt_s_delta, model_point, PC_nocs

    def build_params(self, training_stage_freeze=None):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []

        if 'pose' in training_stage_freeze:
            for param in zip(self.pose_net.parameters()):
                with torch.no_grad():
                    param.requires_grad = False

        # pose
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, self.pose_net.parameters()),
                "lr": float(FLAGS.lr) * FLAGS.lr_pose,
            }
        )

        return params_lr_list

