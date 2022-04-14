import torch

def get_plane(pc, pc_w):
    # min least square
    n = pc.shape[0]
    A = torch.cat([pc[:, :2], torch.ones([n, 1], device=pc.device)], dim=-1)
    b = pc[:, 2].view(-1, 1)
    W = torch.diag(pc_w)
    WA = torch.mm(W, A)
    ATWA = torch.mm(A.permute(1, 0), WA)
    ATWA_1 = torch.inverse(ATWA)
    Wb = torch.mm(W, b)
    ATWb = torch.mm(A.permute(1, 0), Wb)
    X = torch.mm(ATWA_1, ATWb)
    # return dn
    dn_up = torch.cat([X[0] * X[2], X[1] * X[2], -X[2]], dim=0),
    dn_norm = X[0] * X[0] + X[1] * X[1] + 1.0
    dn = dn_up[0] / dn_norm

    normal_n = dn / torch.norm(dn)
    for_p2plane = X[2] / torch.sqrt(dn_norm)
    return normal_n, dn, for_p2plane

def get_plane_parameter(pc, pc_w):
    # min least square
    n = pc.shape[0]
    A = torch.cat([pc[:, :2], torch.ones([n, 1], device=pc.device)], dim=-1)
    b = pc[:, 2].view(-1, 1)
    W = torch.diag(pc_w)
    WA = torch.mm(W, A)
    ATWA = torch.mm(A.permute(1, 0), WA)
    ATWA_1 = torch.inverse(ATWA)
    Wb = torch.mm(W, b)
    ATWb = torch.mm(A.permute(1, 0), Wb)
    X = torch.mm(ATWA_1, ATWb)
    return X