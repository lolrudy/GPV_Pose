import torch
import torch.nn.functional as F
import numpy as np
import absl.flags as flags

FLAGS = flags.FLAGS

def PC_sample(obj_mask, Depth, camK, coor2d):
    '''
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :return:
    '''
    # handle obj_mask
    if obj_mask.shape[1] == 2:   # predicted mask
        obj_mask = F.softmax(obj_mask, dim=1)
        _, obj_mask = torch.max(obj_mask, dim=1)
    '''
    import matplotlib.pyplot as plt
    plt.imshow(obj_mask[0, ...].detach().cpu().numpy())
    plt.show()
    '''
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]

    rand_num = FLAGS.random_points
    samplenum = rand_num

    PC = torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=Depth.device)

    for i in range(bs):
        dp_now = Depth[i, ...].squeeze()   # 256 x 256
        x_now = x_label[i, ...]   # 256 x 256
        y_now = y_label[i, ...]
        obj_mask_now = obj_mask[i, ...].squeeze()  # 256 x 256
        dp_mask = (dp_now > 0.0)
        fuse_mask = obj_mask_now.float() * dp_mask.float()

        camK_now = camK[i, ...]

        # analyze camK
        fx = camK_now[0, 0]
        fy = camK_now[1, 1]
        ux = camK_now[0, 2]
        uy = camK_now[1, 2]

        x_now = (x_now - ux) * dp_now / fx
        y_now = (y_now - uy) * dp_now / fy

        p_n_now = torch.cat([x_now[fuse_mask > 0].view(-1, 1),
                             y_now[fuse_mask > 0].view(-1, 1),
                             dp_now[fuse_mask > 0].view(-1, 1)], dim=1)

        # basic sampling
        if FLAGS.sample_method == 'basic':
            l_all = p_n_now.shape[0]
            if l_all <= 1.0:
                return None, None
            if l_all >= samplenum:
                replace_rnd = False
            else:
                replace_rnd = True

            choose = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
            p_select = p_n_now[choose, :]
        else:
            p_select = None
            raise NotImplementedError

        # reprojection
        if p_select.shape[0] > samplenum:
            p_select = p_select[p_select.shape[0]-samplenum:p_select.shape[0], :]

        PC[i, ...] = p_select[:, :3]

    return PC / 1000.0

def PC_sample_v2(Sketch, obj_mask, Depth, camK, coor2d, seman_feature, nocs_coord=None):
    '''
    :param Sketch: bs x 6 x h x w : each point support each face
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :param rgb_feature
    :return:
    '''
    # handle obj_mask
    if obj_mask.shape[1] == 2:   # predicted mask
        obj_mask = F.softmax(obj_mask, dim=1)
        _, obj_mask = torch.max(obj_mask, dim=1)
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    seman_dim = seman_feature.shape[-1]

    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]

    rand_num = FLAGS.random_points
    samplenum = rand_num

    if nocs_coord is not None:
        nocs_coord = nocs_coord.permute(0,2,3,1)
        PC_nocs = torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=Depth.device)
    else:
        PC_nocs = None

    PC = torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=Depth.device)
    PC_sk = torch.zeros([bs, samplenum, 6], dtype=torch.float32, device=Depth.device)
    PC_seman = torch.zeros([bs, samplenum, seman_dim], dtype=torch.float32, device=Depth.device)

    for i in range(bs):
        sk_now = Sketch[i, ...].squeeze().permute(1, 2, 0)   # 256x256 x 6
        dp_now = Depth[i, ...].squeeze()   # 256 x 256
        x_now = x_label[i, ...]   # 256 x 256
        y_now = y_label[i, ...]
        obj_mask_now = obj_mask[i, ...].squeeze()  # 256 x 256
        dp_mask = (dp_now > 0.0)
        fuse_mask = obj_mask_now.float() * dp_mask.float()
        seman_feature_now = seman_feature[i, ...]
        if nocs_coord is not None:
            nocs_coord_now = nocs_coord[i, ...]
        # sk_now should coorespond to pixels with avaliable depth

        camK_now = camK[i, ...]

        # analyze camK
        fx = camK_now[0, 0]
        fy = camK_now[1, 1]
        ux = camK_now[0, 2]
        uy = camK_now[1, 2]

        x_now = (x_now - ux) * dp_now / fx
        y_now = (y_now - uy) * dp_now / fy

        p_n_now = torch.cat([x_now[fuse_mask > 0].view(-1, 1),
                             y_now[fuse_mask > 0].view(-1, 1),
                             dp_now[fuse_mask > 0].view(-1, 1)], dim=1)
        p_n_sk = sk_now[fuse_mask.bool(), :]  # nn x 6
        p_seman = seman_feature_now[fuse_mask.bool(), :]
        if nocs_coord is not None:
            p_nocs = nocs_coord_now[fuse_mask.bool(), :]
            p_n_f_now = torch.cat([p_n_now, p_n_sk, p_seman, p_nocs], dim=1)
        else:
            p_n_f_now = torch.cat([p_n_now, p_n_sk, p_seman], dim=1)

        # basic sampling
        if FLAGS.sample_method == 'basic':
            l_all = p_n_now.shape[0]
            if l_all <= 1.0:
                return None, None
            if l_all >= samplenum:
                replace_rnd = False
            else:
                replace_rnd = True

            choose = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
            p_select = p_n_f_now[choose, :]
        elif FLAGS.sample_method == 'balance':
            p_n_sk_clone = p_n_sk.clone()
            # sample order, 1, 6, 2, 3, 4, 5
            # y+ 1, y- 6, x + 2, z+ 3, x- 4, z- 5
            snum_f = sup_num // 6
            _, choose_1 = torch.topk(p_n_sk_clone[:, 0], k=snum_f, largest=True)
            p_n_sk_clone[choose_1, :] = 0.0
            _, choose_6 = torch.topk(p_n_sk_clone[:, 5], k=snum_f, largest=True)
            p_n_sk_clone[choose_6, :] = 0.0
            _, choose_2 = torch.topk(p_n_sk_clone[:, 1], k=snum_f, largest=True)
            p_n_sk_clone[choose_2, :] = 0.0
            _, choose_3 = torch.topk(p_n_sk_clone[:, 2], k=snum_f, largest=True)
            p_n_sk_clone[choose_3, :] = 0.0
            _, choose_4 = torch.topk(p_n_sk_clone[:, 3], k=snum_f, largest=True)
            p_n_sk_clone[choose_4, :] = 0.0
            _, choose_5 = torch.topk(p_n_sk_clone[:, 4], k=snum_f, largest=True)
            p_n_sk_clone[choose_5, :] = 0.0
            choose = torch.cat([choose_1, choose_2, choose_3, choose_4, choose_5, choose_6], dim=-1)
            p_select_face = p_n_f_now[choose, :]

            p_n_f_remain = p_n_f_now[torch.sum(p_n_sk_clone, dim=-1) > 0.0, :]  # l_re x 6
            l_re = p_n_f_remain.shape[0]
            choose_r = np.random.choice(l_re, rand_num, replace=True)
            p_select_rand = p_n_f_remain[choose_r, :]

            p_select = torch.cat([p_select_face, p_select_rand], dim=0)
        elif FLAGS.sample_method == 'b+b':  # branch and bound
            p_n_sk_clone = p_n_sk.clone()
            snum_f = FLAGS.per_face_n_of_N
            num_per_f = sup_num // 6

            _, choose = torch.topk(p_n_sk_clone[:, 0], k=snum_f, largest=True)
            choose_rnd = np.random.choice(snum_f, num_per_f, replace=False)
            choose_1 = choose[choose_rnd]
            p_n_sk_clone[choose_1, :] = 0.0

            _, choose = torch.topk(p_n_sk_clone[:, 5], k=snum_f, largest=True)
            choose_rnd = np.random.choice(snum_f, num_per_f, replace=False)
            choose_6 = choose[choose_rnd]
            p_n_sk_clone[choose_6, :] = 0.0

            _, choose = torch.topk(p_n_sk_clone[:, 1], k=snum_f, largest=True)
            choose_rnd = np.random.choice(snum_f, num_per_f, replace=False)
            choose_2 = choose[choose_rnd]
            p_n_sk_clone[choose_2, :] = 0.0

            _, choose = torch.topk(p_n_sk_clone[:, 2], k=snum_f, largest=True)
            choose_rnd = np.random.choice(snum_f, num_per_f, replace=False)
            choose_3 = choose[choose_rnd]
            p_n_sk_clone[choose_3, :] = 0.0

            _, choose = torch.topk(p_n_sk_clone[:, 3], k=snum_f, largest=True)
            choose_rnd = np.random.choice(snum_f, num_per_f, replace=False)
            choose_4 = choose[choose_rnd]
            p_n_sk_clone[choose_4, :] = 0.0

            _, choose = torch.topk(p_n_sk_clone[:, 4], k=snum_f, largest=True)
            choose_rnd = np.random.choice(snum_f, num_per_f, replace=False)
            choose_5 = choose[choose_rnd]
            p_n_sk_clone[choose_5, :] = 0.0

            choose = torch.cat([choose_1, choose_2, choose_3, choose_4, choose_5, choose_6], dim=-1)
            p_select_face = p_n_f_now[choose, :]

            p_n_f_remain = p_n_f_now[torch.sum(p_n_sk_clone, dim=-1) > 0.0, :]  # l_re x 6
            l_re = p_n_f_remain.shape[0]
            choose_r = np.random.choice(l_re, rand_num, replace=True)
            p_select_rand = p_n_f_remain[choose_r, :]

            p_select = torch.cat([p_select_face, p_select_rand], dim=0)
        else:
            p_select = None
            raise NotImplementedError

        # reprojection
        if p_select.shape[0] > samplenum:
            p_select = p_select[p_select.shape[0]-samplenum:p_select.shape[0], :]

        '''
        p_select_x = (p_select[:, 0] - ux) * p_select[:, 2] / fx
        p_select[:, 0] = p_select_x
        p_select_y = (p_select[:, 1] - uy) * p_select[:, 2] / fy
        p_select[:, 1] = p_select_y
        '''

        PC[i, ...] = p_select[:, :3]
        PC_sk[i, ...] = p_select[:, 3:9]
        if nocs_coord is not None:
            PC_nocs[i, ...] = p_select[:, -3:]
            PC_seman[i, ...] = p_select[:, 9:-3]
        else:
            PC_seman[i, ...] = p_select[:, 9:]

    return PC / 1000.0, PC_sk, PC_seman, PC_nocs