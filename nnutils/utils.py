# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import torch


def to_cuda(datapoint):
    skip = ['index']
    for key in datapoint:
        if key in skip:
            continue
        if isinstance(datapoint[key], list):
            datapoint[key] = [e.cuda() for e in datapoint[key]]
        else:
            if hasattr(datapoint[key], 'cuda'):
                datapoint[key] = datapoint[key].cuda()
    return datapoint


def get_model_name(FLAGS):
    # dataset
    name = '%s/%s' % (FLAGS.exp, FLAGS.dataset)
    if FLAGS.dataset.startswith('oi'):
        name += '_%s%g' % (FLAGS.filter_model, FLAGS.filter_trunc,)

    if FLAGS.know_pose == 1:
        name += '_pose'
    if FLAGS.know_mean == 1:
        name += '_3d%d' % FLAGS.vox_loss

    # model
    name += '_%s' % (FLAGS.batch_size)
    name += '_%s' % (FLAGS.g_mod)
    name += '_%s' % (FLAGS.vol_render)

    # loss
    name += '_%s%dm%dc%d' % (FLAGS.mask_loss_type, FLAGS.d_loss_rgb, FLAGS.cyc_mask_loss, FLAGS.content_loss)
    name += '_%s' % (FLAGS.sample_view)
    name += '%d' % FLAGS.seed

    if FLAGS.prior_thin > 0:
        name += 'th%g' % FLAGS.prior_thin
    if FLAGS.prior_blob > 0:
        name += 'bl%g' % FLAGS.prior_blob
    if FLAGS.prior_same > 0:
        name += 'sa%g' % FLAGS.prior_same
    return name


def load_my_state_dict(model: torch.nn.Module, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            # print('Not found in checkpoint', name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[name].size():
            # print('size not match', name, param.size(), own_state[name].size())
            continue
        own_state[name].copy_(param)

