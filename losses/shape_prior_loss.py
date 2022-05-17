import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_distance.chamfer_loss import ChamferLoss


class shape_prior_loss(nn.Module):
    """ Loss for training DeformNet.
        Use NOCS coords to supervise training.
    """
    def __init__(self, corr_wt, cd_wt, entropy_wt, deform_wt, threshold, sym_wt):
        super(shape_prior_loss, self).__init__()
        self.threshold = threshold
        self.chamferloss = ChamferLoss()
        self.corr_wt = corr_wt
        self.cd_wt = cd_wt
        self.entropy_wt = entropy_wt
        self.deform_wt = deform_wt
        self.sym_wt = sym_wt
        self.symmetry_rotation_matrix_list = self.symmetry_rotation_matrix_y()
        self.symmetry_rotation_matrix_list_tensor = None

    def forward(self, assign_mat, deltas, prior, nocs, model, point_mask, sym):
        """
        Args:
            assign_mat: bs x n_pts x nv
            deltas: bs x nv x 3
            prior: bs x nv x 3
        """
        if self.symmetry_rotation_matrix_list_tensor is None:
            result = []
            for rotation_matrix in self.symmetry_rotation_matrix_list:
                rotation_matrix = torch.from_numpy(rotation_matrix).float().to(nocs.device)
                result.append(rotation_matrix)
            self.symmetry_rotation_matrix_list_tensor = result
        loss_dict = {}

        inst_shape = prior + deltas
        # smooth L1 loss for correspondences
        soft_assign = F.softmax(assign_mat, dim=2)
        coords = torch.bmm(soft_assign, inst_shape)  # bs x n_pts x 3
        corr_loss = self.cal_corr_loss(coords, nocs, point_mask, sym)
        corr_loss = self.corr_wt * corr_loss
        # entropy loss to encourage peaked distribution
        log_assign = F.log_softmax(assign_mat, dim=2)
        entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
        entropy_loss = self.entropy_wt * entropy_loss
        # cd-loss for instance reconstruction
        cd_loss, _, _ = self.chamferloss(inst_shape, model)
        cd_loss = self.cd_wt * cd_loss
        # L2 regularizations on deformation
        deform_loss = torch.norm(deltas, p=2, dim=2).mean()
        deform_loss = self.deform_wt * deform_loss
        loss_dict['corr_loss'] = corr_loss          # predicted nocs coordinate loss
        loss_dict['entropy_loss'] = entropy_loss    # entropy loss for assign matrix
        loss_dict['cd_loss'] = cd_loss              # chamfer distance loss between ground truth shape and predicted full shape
        loss_dict['deform_loss'] = deform_loss      # regularization loss for deformation field
        if self.sym_wt != 0:
            loss_dict['sym_loss'] = self.sym_wt * self.cal_sym_loss(inst_shape, sym)
        return loss_dict

    def cal_sym_loss(self, inst_shape, sym):
        # only calculate NOCS in y-axis for symmetric object
        bs = inst_shape.shape[0]
        sym_loss = 0
        if sym is not None:
            target_shape = inst_shape.clone()
            # symmetry aware
            for i in range(bs):
                if sym[i, 0] == 1 and torch.sum(sym[i, 1:]) > 0:  # y axis reflection, can, bowl, bottle
                    target_shape[i, :, 0] = -target_shape[i, :, 0]
                    target_shape[i, :, 2] = -target_shape[i, :, 2]
                elif sym[i, 0] == 0 and sym[i, 1] == 1:  # yx reflection, laptop, mug with handle
                    target_shape[i, :, 2] = -target_shape[i, :, 2]
            sym_loss, _, _ = self.chamferloss(inst_shape, target_shape)
        return sym_loss / bs

    def symmetry_rotation_matrix_y(self, number=30):
        result = []
        for i in range(number):
            theta = 2 * np.pi / number * i
            r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            result.append(r)
        return result

    def cal_corr_loss(self, coords, nocs, point_mask, sym):
        # filter out invalid point
        point_mask = torch.stack([point_mask, point_mask, point_mask], dim=-1)
        coords = torch.where(point_mask, coords, torch.zeros_like(coords))
        nocs = torch.where(point_mask, nocs, torch.zeros_like(nocs))
        # only calculate NOCS in y-axis for symmetric object
        bs = nocs.shape[0]
        corr_loss = 0
        if sym is not None:
            # symmetry aware
            for i in range(bs):
                sym_now = sym[i, 0]
                coords_now = coords[i]
                nocs_now = nocs[i]
                if sym_now == 1:
                    min_corr_loss_now = 1e5
                    min_rotation_matrix = torch.eye(3).cuda()
                    with torch.no_grad():
                        for rotation_matrix in self.symmetry_rotation_matrix_list_tensor:
                            # this should be the inverse of rotation matrix, but it has no influence on result
                            temp_corr_loss = self.cal_corr_loss_for_each_item(coords_now, torch.mm(nocs_now, rotation_matrix))
                            if temp_corr_loss < min_corr_loss_now:
                                min_corr_loss_now = temp_corr_loss
                                min_rotation_matrix = rotation_matrix
                    corr_loss = corr_loss + self.cal_corr_loss_for_each_item(coords_now, torch.mm(nocs_now, min_rotation_matrix))
                else:
                    corr_loss = corr_loss + self.cal_corr_loss_for_each_item(coords_now, nocs_now)
        else:
            for i in range(bs):
                coords_now = coords[i]
                nocs_now = nocs[i]
                corr_loss = corr_loss + self.cal_corr_loss_for_each_item(coords_now, nocs_now)
        return corr_loss / bs

    def cal_corr_loss_for_each_item(self, coords, nocs):
        diff = torch.abs(coords - nocs)
        lower_corr_loss = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher_corr_loss = diff - self.threshold / 2.0
        corr_loss_matrix = torch.where(diff > self.threshold, higher_corr_loss, lower_corr_loss)
        corr_loss_matrix = torch.sum(corr_loss_matrix, dim=-1)
        corr_loss = torch.mean(corr_loss_matrix)
        return corr_loss