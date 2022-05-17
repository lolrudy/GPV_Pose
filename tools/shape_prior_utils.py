import os.path
from tools.rot_utils import get_vertical_rot_vec
import torch

def get_nocs_from_deform(prior, deform_field, assign_mat):
    inst_shape = prior + deform_field
    assign_mat = torch.softmax(assign_mat, dim=2)
    nocs_coords = torch.bmm(assign_mat, inst_shape)
    return nocs_coords


def get_face_dis_from_nocs(nocs_coords, size):
    # y+, x+, z+, x-, z-, y-
    bs, p_num, _ = nocs_coords.shape
    face_dis = torch.zeros((bs, p_num, 6), dtype=nocs_coords.dtype).to(nocs_coords.device)
    for i in range(bs):
        coord_now = nocs_coords[i]
        face_dis_now = face_dis[i]
        size_now = size[i]
        s_x, s_y, s_z = size_now
        diag_len = torch.norm(size_now)
        s_x_norm, s_y_norm, s_z_norm = s_x / diag_len, s_y / diag_len, s_z / diag_len
        face_dis_now[:, 0] = (s_y_norm / 2 - coord_now[:, 1]) * diag_len
        face_dis_now[:, 1] = (s_x_norm / 2 - coord_now[:, 0]) * diag_len
        face_dis_now[:, 2] = (s_z_norm / 2 - coord_now[:, 2]) * diag_len
        face_dis_now[:, 3] = (s_x_norm / 2 + coord_now[:, 0]) * diag_len
        face_dis_now[:, 4] = (s_z_norm / 2 + coord_now[:, 2]) * diag_len
        face_dis_now[:, 5] = (s_y_norm / 2 + coord_now[:, 1]) * diag_len
    return face_dis

def get_face_shift_from_dis(face_dis, rot_y, rot_x, f_y, f_x, use_rectify_normal=False):
    # y+, x+, z+, x-, z-, y-
    bs, p_num, _ = face_dis.shape
    face_shift = torch.zeros((bs, p_num, 18), dtype=face_dis.dtype).to(face_dis.device)
    face_dis = face_dis.unsqueeze(-1)
    for i in range(bs):
        dis_now = face_dis[i]
        face_shift_now = face_shift[i]
        if use_rectify_normal:
            rot_y_now, rot_x_now = get_vertical_rot_vec(f_y[i], f_x[i], rot_y[i, ...], rot_x[i, ...])
            rot_z_now = torch.cross(rot_x_now, rot_y_now)
        else:
            rot_y_now = rot_y[i]
            rot_x_now = rot_x[i]
            rot_z_now = torch.cross(rot_x_now, rot_y_now)
        face_shift_now[:, 0:3] = dis_now[:, 0] * rot_y_now
        face_shift_now[:, 3:6] = dis_now[:, 1] * rot_x_now
        face_shift_now[:, 6:9] = dis_now[:, 2] * rot_z_now
        face_shift_now[:, 9:12] = - dis_now[:, 3] * rot_x_now
        face_shift_now[:, 12:15] = - dis_now[:, 4] * rot_z_now
        face_shift_now[:, 15:18] = - dis_now[:, 5] * rot_y_now
    return face_shift

def get_point_depth_error(nocs_coords, PC, R, t, gt_s, model=None, nocs_scale=None, save_path=None):
    bs, p_num, _ = nocs_coords.shape

    diag_len = torch.norm(gt_s, dim=1)
    diag_len_ = diag_len.view(bs, 1)
    diag_len_ = diag_len_.repeat(1, p_num).view(bs, p_num, 1)
    coords = torch.mul(nocs_coords, diag_len_)
    coords = torch.bmm(R, coords.permute(0, 2, 1)) + t.view(bs, 3, 1)
    coords = coords.permute(0,2,1)
    distance = torch.norm(coords - PC, dim=2, p=1)

    coords_s = torch.mul(nocs_coords, diag_len_)
    pc_proj = torch.bmm(R.permute(0, 2, 1), (PC.permute(0, 2, 1) - t.view(bs, 3, 1))).permute(0, 2, 1)
    nocs_distance = torch.norm(coords_s - pc_proj, dim=2, p=1)

    # assert save_path is None
    if save_path is not None:
        if nocs_scale is None:
            nocs_scale = diag_len
        for i in range(bs):
            pc_now = PC[i]
            coords_now = coords[i]
            pc_proj_now = pc_proj[i]
            coords_s_now = coords_s[i]
            coords_ori_now = nocs_coords[i]
            pc_np = pc_now.detach().cpu().numpy()
            coord_np = coords_now.detach().cpu().numpy()
            pc_proj_np = pc_proj_now.detach().cpu().numpy()
            coords_s_np = coords_s_now.detach().cpu().numpy()
            coords_ori_np = coords_ori_now.detach().cpu().numpy()
            R_np = R[i].detach().cpu().numpy()
            t_np = t[i].detach().cpu().numpy()
            s_np = gt_s[i].detach().cpu().numpy()
            model_np = model[i].detach().cpu().numpy() * nocs_scale[i].detach().cpu().numpy()


            import numpy as np
            np.savetxt(save_path + f'_{i}_pc.txt', pc_np)
            np.savetxt(save_path + f'_{i}_coord2pc.txt', coord_np)
            np.savetxt(save_path + f'_{i}_pc2nocs.txt', pc_proj_np)
            np.savetxt(save_path + f'_{i}_coord.txt', coords_s_np)
            np.savetxt(save_path + f'_{i}_coord_ori.txt', coords_ori_np)
            np.savetxt(save_path + f'_{i}_model.txt', model_np)
            np.savetxt(save_path + f'_{i}_r.txt', R_np)
            np.savetxt(save_path + f'_{i}_t.txt', t_np)
            np.savetxt(save_path + f'_{i}_s.txt', s_np)
    return distance, nocs_distance

def get_nocs_model(model_point):
    model_point_nocs = torch.zeros_like(model_point).to(model_point.device)
    bs = model_point.shape[0]
    for i in range(bs):
        model_now = model_point[i]
        lx = 2 * torch.max(torch.max(model_now[:, 0]), -torch.min(model_now[:, 0]))
        ly = torch.max(model_now[:, 1]) - torch.min(model_now[:, 1])
        lz = torch.max(model_now[:, 2]) - torch.min(model_now[:, 2])
        diagonal_len = torch.norm(torch.tensor([lx, ly, lz]))
        print('model diagonal in final', diagonal_len)
        model_point_nocs[i, :] = model_now / diagonal_len
    return model_point_nocs