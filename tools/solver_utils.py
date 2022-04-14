# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List

import torch
from detectron2.config import CfgNode
from detectron2.solver import WarmupCosineLR, WarmupMultiStepLR
from tools.torch_utils.solver.lr_scheduler import flat_and_anneal_lr_scheduler
from tools.torch_utils.solver.ranger2020 import Ranger
import absl.flags as flags
FLAGS = flags.FLAGS

__all__ = ["build_lr_scheduler", "build_optimizer_with_params"]

'''
def register_optimizer(name):
    """TODO: add more optimizers"""
    if name in OPTIMIZERS:
        return
    if name == "Ranger":
        from tools.torch_utils.solver.ranger import Ranger

        # from lib.torch_utils.solver.ranger2020 import Ranger
        OPTIMIZERS.register_module()(Ranger)
    elif name in ["AdaBelief", "RangerAdaBelief"]:
        from tools.torch_utils.solver.AdaBelief import AdaBelief
        from tools.torch_utils.solver.ranger_adabelief import RangerAdaBelief

        OPTIMIZERS.register_module()(AdaBelief)
        OPTIMIZERS.register_module()(RangerAdaBelief)
    elif name in ["SGDP", "AdamP"]:
        from tools.torch_utils.solver.adamp import AdamP
        from tools.torch_utils.solver.sgdp import SGDP

        OPTIMIZERS.register_module()(AdamP)
        OPTIMIZERS.register_module()(SGDP)
    elif name in ["SGD_GC", "SGD_GCC"]:
        from tools.torch_utils.solver.sgd_gc import SGD_GC, SGD_GCC

        OPTIMIZERS.register_module()(SGD_GC)
        OPTIMIZERS.register_module()(SGD_GCC)
    else:
        raise ValueError(f"Unknown optimizer name: {name}")
'''
# note that this is adapted from mmcv, if you dont want to use ranger,
# please use the provieded build from cfg in mmcv
def build_optimizer_with_params(cfg, params):
    if cfg.SOLVER.OPTIMIZER_CFG == "":
        raise RuntimeError("please provide cfg.SOLVER.OPTIMIZER_CFG to build optimizer")
    if cfg.SOLVER.OPTIMIZER_CFG.type.lower() == "ranger":
        return Ranger(params=params, lr=cfg.SOLVER.OPTIMIZER_CFG.lr, weight_decay=cfg.SOLVER.OPTIMIZER_CFG.weight_decay)
    else:
        return None

def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer, total_iters: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a LR scheduler from config."""
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    steps = [rel_step * total_iters for rel_step in cfg.SOLVER.REL_STEPS]
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            steps,  # cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            total_iters,  # cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name.lower() == "flat_and_anneal":
        return flat_and_anneal_lr_scheduler(
            optimizer,
            total_iters=total_iters,  # NOTE: TOTAL_EPOCHS * len(train_loader)
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,  # default "linear"
            anneal_method=cfg.SOLVER.ANNEAL_METHOD,
            anneal_point=cfg.SOLVER.ANNEAL_POINT,  # default 0.72
            steps=cfg.SOLVER.get("REL_STEPS", [2 / 3.0, 8 / 9.0]),  # default [2/3., 8/9.], relative decay steps
            target_lr_factor=cfg.SOLVER.get("TARTGET_LR_FACTOR", 0),
            poly_power=cfg.SOLVER.get("POLY_POWER", 1.0),
            step_gamma=cfg.SOLVER.GAMMA,  # default 0.1
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
