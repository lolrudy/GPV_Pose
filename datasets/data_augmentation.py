# introduced from fs-net
import numpy as np
import cv2
import torch
import math


# add noise to mask
def defor_2D(roi_mask, rand_r=2, rand_pro=0.3):
    '''
    :param roi_mask: 256 x 256
    :param rand_r: randomly expand or shrink the mask iter rand_r
    :return:
    '''
    roi_mask = roi_mask.copy().squeeze()
    if np.random.rand() > rand_pro:
        return roi_mask
    mask = roi_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_erode = cv2.erode(mask, kernel, rand_r)  # rand_r
    mask_dilate = cv2.dilate(mask, kernel, rand_r)
    change_list = roi_mask[mask_erode != mask_dilate]
    l_list = change_list.size
    if l_list < 1.0:
        return roi_mask
    choose = np.random.choice(l_list, l_list // 2, replace=False)
    change_list = np.ones_like(change_list)
    change_list[choose] = 0.0
    roi_mask[mask_erode != mask_dilate] = change_list
    roi_mask[roi_mask > 0.0] = 1.0
    return roi_mask


# point cloud based data augmentation
# augment based on bounding box
def defor_3D_bb(pc, R, t, s, nocs, model, sym=None, aug_bb=None):
    # pc  n x 3, here s must  be the original s
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    if sym[0] == 1:  # y axis symmetry
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        exz = (ex + ez) / 2
        pc_reproj[:, (0, 2)] = pc_reproj[:, (0, 2)] * exz
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        nocs_scale_aug = torch.norm(torch.tensor([s[0] * exz, s[1] * ey, s[2] * exz])) / torch.norm(s)
        s[0] = s[0] * exz
        s[1] = s[1] * ey
        s[2] = s[2] * exz
        nocs[:, 0] = nocs[:, 0] * exz / nocs_scale_aug
        nocs[:, 1] = nocs[:, 1] * ey / nocs_scale_aug
        nocs[:, 2] = nocs[:, 2] * exz / nocs_scale_aug
        model[:, 0] = model[:, 0] * exz / nocs_scale_aug
        model[:, 1] = model[:, 1] * ey / nocs_scale_aug
        model[:, 2] = model[:, 2] * exz / nocs_scale_aug
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
    else:
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]
        nocs_scale_aug = torch.norm(torch.tensor([s[0] * ex, s[1] * ey, s[2] * ez])) / torch.norm(s)
        pc_reproj[:, 0] = pc_reproj[:, 0] * ex
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        pc_reproj[:, 2] = pc_reproj[:, 2] * ez
        s[0] = s[0] * ex
        s[1] = s[1] * ey
        s[2] = s[2] * ez
        nocs[:, 0] = nocs[:, 0] * ex / nocs_scale_aug
        nocs[:, 1] = nocs[:, 1] * ey / nocs_scale_aug
        nocs[:, 2] = nocs[:, 2] * ez / nocs_scale_aug
        model[:, 0] = model[:, 0] * ex / nocs_scale_aug
        model[:, 1] = model[:, 1] * ey / nocs_scale_aug
        model[:, 2] = model[:, 2] * ez / nocs_scale_aug
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
    return pc_new, s, nocs, model


def defor_3D_bc(pc, R, t, s, model_point, nocs_scale, nocs):
    # resize box cage along y axis, the size s is modified
    ey_up = torch.rand(1, device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand(1,  device=pc.device) * (1.2 - 0.8) + 0.8
    # for each point, resize its x and z linealy
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = (pc_reproj[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    pc_reproj[:, 0] = pc_reproj[:, 0] * per_point_resize
    pc_reproj[:, 2] = pc_reproj[:, 2] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    norm_s = s / torch.norm(s)
    model_point_resize =  (model_point[:, 1] + norm_s[1] / 2) / norm_s[1] * (ey_up - ey_down) + ey_down
    model_point[:, 0] = model_point[:, 0] * model_point_resize
    model_point[:, 2] = model_point[:, 2] * model_point_resize

    lx = 2 * max(max(model_point[:, 0]), -min(model_point[:, 0]))
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * torch.norm(s)
    ly_t = ly * torch.norm(s)
    lz_t = lz * torch.norm(s)
    size_new = torch.tensor([lx_t, ly_t, lz_t], device=pc.device)

    nocs_scale_aug = torch.norm(torch.tensor([lx, ly, lz]))
    model_point = model_point / nocs_scale_aug

    nocs_resize = (nocs[:, 1] + norm_s[1] / 2) / norm_s[1] * (ey_up - ey_down) + ey_down
    nocs[:, 0] = nocs[:, 0] * nocs_resize
    nocs[:, 2] = nocs[:, 2] * nocs_resize
    nocs = nocs / nocs_scale_aug

    return pc_new, size_new, model_point, nocs


# point cloud based data augmentation
# augment based on bounding box
def deform_non_linear(pc, R, t, s, nocs, model_point, axis=0):
    # pc  n x 3, here s must  be the original s
    assert axis == 0
    r_max = torch.rand(1, device=pc.device) * 0.2 + 1.1
    r_min = - torch.rand(1, device=pc.device) * 0.2 + 0.9
    # for each point, resize its x and z
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = r_max - 4 * (pc_reproj[:, 1] * pc_reproj[:, 1]) / (s[1] ** 2) * (r_max - r_min)
    pc_reproj[:, 0] = pc_reproj[:, 0] * per_point_resize
    pc_reproj[:, 2] = pc_reproj[:, 2] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    norm_s = s / torch.norm(s)
    model_point_resize = (model_point[:, 1] + norm_s[1] / 2) / norm_s[1] * (r_max - r_min) + r_min
    model_point[:, 0] = model_point[:, 0] * model_point_resize
    model_point[:, 2] = model_point[:, 2] * model_point_resize

    lx = 2 * max(max(model_point[:, 0]), -min(model_point[:, 0]))
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * torch.norm(s)
    ly_t = ly * torch.norm(s)
    lz_t = lz * torch.norm(s)
    size_new = torch.tensor([lx_t, ly_t, lz_t], device=pc.device)

    nocs_scale_aug = torch.norm(torch.tensor([lx, ly, lz]))
    model_point = model_point / nocs_scale_aug

    nocs_resize = (nocs[:, 1] + norm_s[1] / 2) / norm_s[1] * (r_max - r_min) + r_min
    nocs[:, 0] = nocs[:, 0] * nocs_resize
    nocs[:, 2] = nocs[:, 2] * nocs_resize
    nocs = nocs / nocs_scale_aug
    return pc_new, size_new, model_point, nocs


def defor_3D_pc(pc, r):
    points_defor = torch.randn(pc.shape).to(pc.device)
    pc = pc + points_defor * r
    return pc


# point cloud based data augmentation
# random rotation and translation
def defor_3D_rt(pc, R, t, aug_rt_t, aug_rt_r):
    #  add_t
    dx = aug_rt_t[0]
    dy = aug_rt_t[1]
    dz = aug_rt_t[2]

    pc[:, 0] = pc[:, 0] + dx
    pc[:, 1] = pc[:, 1] + dy
    pc[:, 2] = pc[:, 2] + dz
    t[0] = t[0] + dx
    t[1] = t[1] + dy
    t[2] = t[2] + dz

    # add r
    '''
    Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
    Rm_tensor = torch.tensor(Rm, device=pc.device)
    pc_new = torch.mm(Rm_tensor, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm_tensor, R)
    R = R_new
    '''
    '''
    x_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    y_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    z_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    Rm = get_rotation_torch(x_rot, y_rot, z_rot)
    '''
    Rm = aug_rt_r
    pc_new = torch.mm(Rm, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm, R)
    R = R_new
    T_new = torch.mm(Rm, t.view(3, 1))
    t = T_new

    return pc, R, t


def get_rotation(x_, y_, z_):
    # print(math.cos(math.pi/2))
    x = float(x_ / 180) * math.pi
    y = float(y_ / 180) * math.pi
    z = float(z_ / 180) * math.pi
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]])

    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x)).astype(np.float32)

def get_rotation_torch(x_, y_, z_):
    x = (x_ / 180) * math.pi
    y = (y_ / 180) * math.pi
    z = (z_ / 180) * math.pi
    R_x = torch.tensor([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]], device=x_.device)

    R_y = torch.tensor([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]], device=y_.device)

    R_z = torch.tensor([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]], device=z_.device)
    return torch.mm(R_z, torch.mm(R_y, R_x))
