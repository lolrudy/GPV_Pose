import torch
import numpy as np
import absl.flags as flags

FLAGS = flags.FLAGS
from mmcv import Config
from tools.solver_utils import build_lr_scheduler, build_optimizer_with_params


# important parameters used here
# total_iters: total_epoch x iteration per epoch

def build_lr_rate(optimizer, total_iters):
    # build cfg from flags
    cfg = dict(
        SOLVER=dict(
            IMS_PER_BATCH=FLAGS.batch_size,
            TOTAL_EPOCHS=FLAGS.total_epoch,
            LR_SCHEDULER_NAME=FLAGS.lr_scheduler_name,
            REL_STEPS=(0.5, 0.75),
            ANNEAL_METHOD=FLAGS.anneal_method,  # "cosine"
            ANNEAL_POINT=FLAGS.anneal_point,
            # REL_STEPS=(0.3125, 0.625, 0.9375),
            OPTIMIZER_CFG=dict(type=FLAGS.optimizer_type, lr=FLAGS.lr, weight_decay=0),
            WEIGHT_DECAY=FLAGS.weight_decay,
            WARMUP_FACTOR=FLAGS.warmup_factor,
            WARMUP_ITERS=FLAGS.warmup_iters,
            WARMUP_METHOD=FLAGS.warmup_method,
            GAMMA=FLAGS.gamma,
            POLY_POWER=FLAGS.poly_power,
        ),
    )
    cfg = Config(cfg)
    scheduler = build_lr_scheduler(cfg, optimizer, total_iters=total_iters)
    return scheduler


def build_optimizer(params):
    # build cfg from flags
    cfg = dict(
        SOLVER=dict(
            IMS_PER_BATCH=FLAGS.batch_size,
            TOTAL_EPOCHS=FLAGS.total_epoch,
            LR_SCHEDULER_NAME=FLAGS.lr_scheduler_name,
            ANNEAL_METHOD=FLAGS.anneal_method,  # "cosine"
            ANNEAL_POINT=FLAGS.anneal_point,
            # REL_STEPS=(0.3125, 0.625, 0.9375),
            OPTIMIZER_CFG=dict(type=FLAGS.optimizer_type, lr=FLAGS.lr, weight_decay=0),
            WEIGHT_DECAY=FLAGS.weight_decay,
            WARMUP_FACTOR=FLAGS.warmup_factor,
            WARMUP_ITERS=FLAGS.warmup_iters,
        ),
    )
    cfg = Config(cfg)
    optimizer = build_optimizer_with_params(cfg, params)
    return optimizer


def get_gt_v(Rs, axis=2):
    bs = Rs.shape[0]  # bs x 3 x 3
    # TODO use 3 axis, the order remains: do we need to change order?
    if axis == 3:
        corners = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float).to(Rs.device)
        corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
        gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
    else:
        assert axis == 2
        corners = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.float).to(Rs.device)
        corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
        gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
    gt_green = gt_vec[:, 3:6]
    gt_red = gt_vec[:, (6, 7, 8)]
    return gt_green, gt_red
