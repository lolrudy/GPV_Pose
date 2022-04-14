import os
import random

import mmcv
import torch
from absl import app

from config.config import *
from tools.training_utils import build_lr_rate, get_gt_v, build_optimizer
from network.GPVPose import GPVPose

FLAGS = flags.FLAGS
from datasets.load_data import PoseDataset
import numpy as np
import time

# from creating log
import tensorflow as tf
from tools.eval_utils import setup_logger, compute_sRT_errors
torch.autograd.set_detect_anomaly(True)
device = 'cuda'

def train(argv):
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    tf.compat.v1.disable_eager_execution()
    tb_writter = tf.compat.v1.summary.FileWriter(FLAGS.model_save)
    logger = setup_logger('train_log', os.path.join(FLAGS.model_save, 'log.txt'))
    for key, value in vars(FLAGS).items():
        logger.info(key + ':' + str(value))
    Train_stage = 'PoseNet_only'
    network = GPVPose(Train_stage)
    network = network.to(device)
    # resume or not
    if FLAGS.resume:
        network.load_state_dict(torch.load(FLAGS.resume_model))
        s_epoch = FLAGS.resume_point
    else:
        s_epoch = 0

    # build dataset annd dataloader
    train_dataset = PoseDataset(source=FLAGS.dataset, mode='train',
                                data_dir=FLAGS.dataset_dir, per_obj=FLAGS.per_obj)
    # start training datasets sampler
    st_time = time.time()
    train_steps = FLAGS.train_steps
    global_step = train_steps * s_epoch  # record the number iteration
    train_size = train_steps * FLAGS.batch_size
    indices = []
    page_start = - train_size

    #  build optimizer
    param_list = network.build_params(training_stage_freeze=[])
    optimizer = build_optimizer(param_list)
    optimizer.zero_grad()   # first clear the grad
    scheduler = build_lr_rate(optimizer, total_iters=train_steps * FLAGS.total_epoch // FLAGS.accumulate)
    #  training iteration, this code is develop based on object deform net
    for epoch in range(s_epoch, FLAGS.total_epoch):
        # train one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                      ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
        # create optimizer and adjust learning rate accordingly
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if FLAGS.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len + real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // (4 * real_len) + 1
                data_list = random.sample(camera_indices, 3 * n_repeat * real_len) + real_indices * n_repeat
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start + train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=FLAGS.num_workers, pin_memory=True)
        network.train()

        #################################
        for i, data in enumerate(train_dataloader, 1):

            output_dict, loss_dict \
                = network(rgb=data['roi_img'].to(device), depth=data['roi_depth'].to(device),
                          depth_normalize=data['depth_normalize'].to(device),
                          obj_id=data['cat_id'].to(device), camK=data['cam_K'].to(device), gt_mask=data['roi_mask'].to(device),
                          gt_R=data['rotation'].to(device), gt_t=data['translation'].to(device),
                          gt_s=data['fsnet_scale'].to(device), mean_shape=data['mean_shape'].to(device),
                          gt_2D=data['roi_coord_2d'].to(device), sym=data['sym_info'].to(device),
                          aug_bb=data['aug_bb'].to(device), aug_rt_t=data['aug_rt_t'].to(device), aug_rt_r=data['aug_rt_R'].to(device),
                          def_mask=data['roi_mask_deform'].to(device),
                          model_point=data['model_point'].to(device), nocs_scale=data['nocs_scale'].to(device), do_loss=True)
            fsnet_loss = loss_dict['fsnet_loss']
            recon_loss = loss_dict['recon_loss']
            geo_loss = loss_dict['geo_loss']
            prop_loss = loss_dict['prop_loss']

            total_loss = sum(fsnet_loss.values()) + sum(recon_loss.values()) \
                            + sum(geo_loss.values()) + sum(prop_loss.values()) \

            # backward
            if global_step % FLAGS.accumulate == 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)

            global_step += 1
            summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='lr',
                                                                             simple_value=optimizer.param_groups[0]["lr"]),
                                                  tf.compat.v1.Summary.Value(tag='train_loss', simple_value=total_loss),
                                                  tf.compat.v1.Summary.Value(tag='rot_loss_1',
                                                                             simple_value=fsnet_loss['Rot1']),
                                                  tf.compat.v1.Summary.Value(tag='rot_loss_2',
                                                                             simple_value=fsnet_loss['Rot2']),
                                                  tf.compat.v1.Summary.Value(tag='T_loss',
                                                                             simple_value=fsnet_loss['Tran']),
                                                  tf.compat.v1.Summary.Value(tag='Prop_sym_recon',
                                                                             simple_value=prop_loss['Prop_sym_recon']),
                                                  tf.compat.v1.Summary.Value(tag='Prop_sym_rt',
                                                                             simple_value=prop_loss['Prop_sym_rt']),
                                                  tf.compat.v1.Summary.Value(tag='Size_loss',
                                                                             simple_value=fsnet_loss['Size']),
                                                  tf.compat.v1.Summary.Value(tag='Face_loss',
                                                                             simple_value=recon_loss['recon_per_p']),
                                                  tf.compat.v1.Summary.Value(tag='Recon_loss_r',
                                                                             simple_value=recon_loss['recon_point_r']),
                                                  tf.compat.v1.Summary.Value(tag='Recon_loss_t',
                                                                             simple_value=recon_loss['recon_point_t']),
                                                  tf.compat.v1.Summary.Value(tag='Recon_loss_s',
                                                                             simple_value=recon_loss['recon_point_s']),
                                                  tf.compat.v1.Summary.Value(tag='Recon_p_f',
                                                                             simple_value=recon_loss['recon_p_f']),
                                                  tf.compat.v1.Summary.Value(tag='Recon_loss_se',
                                                                             simple_value=recon_loss['recon_point_self']),
                                                  tf.compat.v1.Summary.Value(tag='Face_loss_vote',
                                                                             simple_value=recon_loss['recon_point_vote']),
                                                  ])
            tb_writter.add_summary(summary, global_step)

            if i % FLAGS.log_every == 0:
                logger.info('Batch {0} Loss:{1:f}, rot_loss:{2:f}, size_loss:{3:f}, trans_loss:{4:f}'.format(
                        i, total_loss.item(), (fsnet_loss['Rot1']+fsnet_loss['Rot2']).item(),
                    fsnet_loss['Size'].item(), fsnet_loss['Tran'].item()))

        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))

        # save model
        if (epoch + 1) % FLAGS.save_every == 0 or (epoch + 1) == FLAGS.total_epoch:
            torch.save(network.state_dict(), '{0}/model_{1:02d}.pth'.format(FLAGS.model_save, epoch))


if __name__ == "__main__":
    app.run(train)
