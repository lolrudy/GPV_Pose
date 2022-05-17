import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from mmcv.cnn import normal_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm
from nnutils.layer_utils import get_norm, get_nn_act_func

FLAGS = flags.FLAGS
# Point_center  encode the segmented point cloud
# one more conv layer compared to original paper

class Pose_Ts(nn.Module):
    def __init__(self):
        super(Pose_Ts, self).__init__()
        self.f = FLAGS.feat_pcl + FLAGS.feat_global_pcl + FLAGS.feat_seman + 3
        self.k = FLAGS.Ts_c

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
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

class Pose_Ts_global(nn.Module):
    def __init__(
        self,
        feat_dim=256,
        num_layers=2,
        norm="none",
        num_gn_groups=32,
        act="leaky_relu",
        num_classes=1,
        norm_input=False,
    ):
        super().__init__()
        in_dim = FLAGS.feat_global_pcl
        self.norm = get_norm(norm, feat_dim, num_gn_groups=num_gn_groups)
        self.act_func = act_func = get_nn_act_func(act)
        self.num_classes = num_classes
        self.linears = nn.ModuleList()
        if norm_input:
            self.linears.append(nn.BatchNorm1d(in_dim))
        for _i in range(num_layers):
            _in_dim = in_dim if _i == 0 else feat_dim
            self.linears.append(nn.Linear(_in_dim, feat_dim))
            self.linears.append(get_norm(norm, feat_dim, num_gn_groups=num_gn_groups))
            self.linears.append(act_func)

        self.fc_t = nn.Linear(feat_dim, 3 * num_classes)
        self.fc_s = nn.Linear(feat_dim, 3 * num_classes)

        # init ------------------------------------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_t, std=0.01)
        normal_init(self.fc_s, std=0.01)

    def forward(self, x):
        """
        x: should be flattened
        """
        for _layer in self.linears:
            x = _layer(x)

        trans = self.fc_t(x)
        scale = self.fc_s(x)
        return trans, scale


def main(argv):
    feature = torch.rand(3, 3, 1000)
    obj_id = torch.randint(low=0, high=15, size=[3, 1])
    net = Pose_Ts()
    out = net(feature, obj_id)
    t = 1

if __name__ == "__main__":
    app.run(main)
