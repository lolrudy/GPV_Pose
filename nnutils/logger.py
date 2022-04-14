# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, model_name, save_dir):
        self.model_name = os.path.basename(model_name)
        self.plotter_dict = {}
        cmd = 'rm -rf %s' % os.path.join(save_dir, model_name)
        os.system(cmd)

        self.save_dir = os.path.join(save_dir, model_name, 'train')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print('## Make Directory: ', self.save_dir)

        cmd = 'rm -rf %s' % self.save_dir
        os.system(cmd)
        print(cmd)
        self.tf_wr = SummaryWriter(self.save_dir)

    def add_loss(self, t, dictionary, pref=''):
        for key in dictionary:
            name = pref + key.replace(':', '/')
            self.tf_wr.add_scalar(name, dictionary[key], t)

    def add_hist_by_dim(self, t, z, name='', max_dim=10):
        dim = z.size(-1)
        dim = min(dim, max_dim)
        for d in range(dim):
            index = name + '/%d' % d
            self.tf_wr.add_histogram(index, z[:, d], t)

    def add_images(self, iteration, images, name=''):
        """
        :param iteration:
        :param images:  Tensor (N, C, H, W), in range (-1, 1)
        :param name:
        :return:
        """
        # images = torch.stack(images, dim=0),
        # x = vutils.make_grid(images)
        images = images.cpu().detach()
        x = vutils.make_grid(images)
        self.tf_wr.add_image(name, x / 2 + 0.5, iteration)

    def print(self, t, epoch, losses, total_loss):
        print('[Epoch %2d] iter: %d of model' % (epoch, t), self.model_name)
        print('\tTotal Loss: %.6f' % total_loss)
        for k in losses:
            print('\t\t%s: %.6f' % (k, losses[k]))