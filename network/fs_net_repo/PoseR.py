import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from config.config import *
FLAGS = flags.FLAGS
from mmcv.cnn import normal_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

class RotHead(nn.Module):
    def __init__(self):
        super(RotHead, self).__init__()
        self.f = FLAGS.feat_pcl + FLAGS.feat_global_pcl + FLAGS.feat_seman
        self.k = FLAGS.R_c

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self._init_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

def main(argv):
    points = torch.rand(2, 1350, 1500)  # batchsize x feature x numofpoint
    rot_head = RotHead()
    rot = rot_head(points)
    t = 1


if __name__ == "__main__":
    app.run(main)
