import torch
import torch.nn as nn
import numpy as np
from tools.rot_utils import get_vertical_rot_vec, get_rot_mat_y_first
def azel2uni(view_para, homo=True):
    """
    :param view_para: tensor in shape of (N, 6 / 2) az, el, scale, x, y, z
    :return: scale: (N, 1), trans: (N, 3). rot: (N, 4, 4)
    """
    if view_para.size(1) == 2:
        az, el = torch.split(view_para, [1, 1], dim=1)
        zeros = torch.zeros_like(az)
        ones = torch.ones_like(az)

        view_para = torch.cat([az, el, ones, zeros, zeros, zeros + 2], dim=1)  # (N, 6)

    az, el, scale, trans = torch.split(view_para, [1, 1, 1, 3], dim=-1)
    rot = azel2rot(az, el, homo)
    return scale, trans, rot


def homo_matrix(rot: torch.Tensor):
    """
    :param rot: (N, 3, 3)
    :return: (N, 4, 4)
    """
    device = rot.device
    N = rot.size(0)
    zeros = torch.zeros([N, 1, 1], device=device)
    rotation_matrix = torch.cat([
        torch.cat([rot, torch.zeros(N, 3, 1, device=device)], dim=2),
        torch.cat([zeros, zeros, zeros, zeros + 1], dim=2)
    ], dim=1)
    return rotation_matrix


def azel2rot(az, el, homo=True):
    """
    :param az: (N, 1, (1)). y-axis
    :param el: x-axis
    :return: rot: (N, 4, 4). rotation: Ry? then Rx? x,y,z
    """
    N = az.size(0)
    az = az.view(N, 1, 1)
    el = el.view(N, 1, 1)
    ones = torch.ones_like(az)
    zeros = torch.zeros_like(az)

    # rot = py_t.euler_angles_to_matrix(torch.cat([az.view(N, 1), el.view(N, 1), zeros.view(N, 1)], dim=1),'YXZ')
    # return rot
    batch_rot_y = torch.cat([
        torch.cat([torch.cos(az), zeros, -torch.sin(az)], dim=2),
        torch.cat([zeros, ones, zeros], dim=2),
        torch.cat([torch.sin(az), zeros, torch.cos(az)], dim=2),
    ], dim=1)

    batch_rot_x = torch.cat([
        torch.cat([ones, zeros, zeros], dim=2),
        torch.cat([zeros, torch.cos(el), torch.sin(el)], dim=2),
        torch.cat([zeros, -torch.sin(el), torch.cos(el)], dim=2),
    ], dim=1)
    rotation_matrix = torch.matmul(batch_rot_y, batch_rot_x)
    if homo:
        rotation_matrix = homo_matrix(rotation_matrix)
    return rotation_matrix

def diag_to_homo(diag):
    """
    :param diag: (N, )
    :return:
    """
    N = diag.size(0)
    diag = diag.view(N, 1, 1)

    zeros = torch.zeros_like(diag)
    ones = torch.ones_like(diag)
    mat = torch.cat([
        torch.cat([diag, zeros, zeros, zeros], dim=2),
        torch.cat([zeros, diag, zeros, zeros], dim=2),
        torch.cat([zeros, zeros, diag, zeros], dim=2),
        torch.cat([zeros, zeros, zeros, ones], dim=2),
    ], dim=1)
    return mat

def homo_to_3x3(rot):
    return rot[:, :3, :3]

def rt_to_homo(rot, t=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :return: (N, 4, 4) [R, t; 0, 1]
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] = 1
    mat = torch.cat([mat, zeros], dim=-2)
    return mat

def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    #print(P.shape,Q.shape)
    # print(np.mean(P,0))
    # P= P-np.mean(P,0)
    # Q =Q - np.mean(Q, 0)
    # print(P)
    # tests
    C = np.dot(P.T, Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    U, S, V = np.linalg.svd(C)
    #S=np.diag(S)
    #print(C)
    # print(S)
    #print(np.dot(U,np.dot(S,V)))
    # d = (np.linalg.det(V.T) * np.linalg.det(U.T)) <0.0

    # d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    # E = np.diag(np.array([1, 1, 1]))
    # if d:
    #     S[-1] = -S[-1]
    #     V[:, -1] = -V[:, -1]
    E = np.diag(np.array([1.0, 1.0, (np.linalg.det(V.T) * np.linalg.det(U.T))], dtype=np.float32))


    # print(E)

    # Create Rotation matrix U
    #print(V)
    #print(U)
    R = np.dot(V.T ,np.dot(E,U.T))

    return R



def gettrans(kps,h):
    hss=[]
    kps=kps.reshape(-1,3)
    for i in range(h.shape[1]):
        P = kps.T - kps.T.mean(1).reshape((3, 1))
        Q = h[:,i,:].T - h[:,i,:].T.mean(1).reshape((3,1))
        P, Q = P.T, Q.T
        R = kabsch(P, Q) ##N*3, N*3
        # T = h[:,i,:]-np.dot(R,kps.T).T
        hh = np.zeros((3, 4), dtype=np.float32)
        hh[0:3,0:3]=R
        # hh[0:3,3]=np.mean(T,0)
        hss.append(hh)
    return hss


def generate_sRT(R, T, s, mode):   #generate sRT mat
    # useless..
    bs = T.shape[0]
    res = generate_RT(R, T, mode)
    for i in range(bs):
        s_now = s[i, ...]  # 3,
        s_nocs = torch.norm(s_now)  # or 1/ s_nocs
        res[i, :3, :3] = s_nocs * res[i, :3, :3]
    return res

#  note that now using vec method, we directly estimate R with damper
def generate_RT(R, f, T, mode, sym):   #generate sRT mat
    bs = T.shape[0]
    res = torch.zeros([bs, 4, 4], dtype=torch.float).to(T.device)
    if mode == "vec":
        # generate from green and red vec
        for i in range(bs):
            if sym[i, 0] == 1:
                c1 = f[0][i]
                c2 = 0
            else:
                c1 = f[0][i]
                c2 = f[1][i]
            pred_green_vec = R[0][i, ...]  # 2 x 3
            pred_red_vec = R[1][i, ...]
            new_y, new_x = get_vertical_rot_vec(c1, c2, pred_green_vec, pred_red_vec)
            p_R = get_rot_mat_y_first(new_y.view(1, -1), new_x.view(1, -1))[0]  # 3 x 3
            RT = np.identity(4)
            RT[:3, :3] = p_R.cpu().numpy()
            RT[:3, 3] = T[i, ...].cpu().numpy()
            res[i, :, :] = torch.from_numpy(RT).to(T.device)
        return res

    elif mode == "gt":  # directly generate sRT from R T s
        for i in range(bs):
            RT = np.identity(4)
            RT[:3, :3] = R[i, ...].cpu().numpy()
            RT[:3, 3] = T[i, ...].cpu().numpy()
            res[i, :, :] = torch.from_numpy(RT).to(T.device)
        return res


def RecoverRtsfromVec(green_R, red_R, T, s, num_cor=3):
    # assume batchsize x 6 x 1, batchsize x 3 x 1
    bs = green_R.shape[0]
    res = torch.zeros([bs, 3, 4])
    if num_cor == 3:
        corners_ = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    else:
        corners_ = np.array([[0, 0, 0], [0, 1, 0]])
    corners_ = corners_ / np.linalg.norm(corners_)
    pred_axis = torch.zeros([num_cor, 3])
    if torch.cuda.is_available():
        corners_ = torch.tensor(corners_).cuda()
        res = res.cuda()
        pred_axis = pred_axis.cuda()
    for ib in range(bs):
        pred_axis[0:2, :] = green_R[ib, :].view((2, 3))
        if num_cor == 3:
            pred_axis[2, :] = red_R[ib, :].view((2, 3))[1, :]
        pred_axis = pred_axis / torch.norm(pred_axis)
        # calibrate the two point cloud and get R
        pose = gettrans(corners_.reshape((num_cor, 3)), pred_axis.reshape((num_cor, 1, 3)))
        R = pose[0][0:3, 0:3]  # R 3 x 3
        res[ib, :, 0:3] = R
        res[ib, :, 3] = T[ib, :]
    return res, s


if __name__ == '__main__':
    from scipy.spatial.transform import Rotation
    R = Rotation.random().as_matrix()
    print(R)
    R = torch.FloatTensor(np.expand_dims(R, 0))
    t = np.array([-0.1, 0.1, 0.9])
    t = torch.FloatTensor(np.expand_dims(t, 0))
    s = np.array([1, 1, 1])
    s = torch.FloatTensor(np.expand_dims(s, 0))

    from tools.training_utils import get_gt_v
    green_R, red_R = get_gt_v(R, axis=3)
    print(green_R, red_R)
    pred_RT = generate_RT([green_R * 2, red_R], t, mode='vec', axis=3)
    print(pred_RT)