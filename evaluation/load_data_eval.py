import os
import cv2
import math
import random
import numpy as np
import _pickle as cPickle
from config.config import *
from datasets.data_augmentation import defor_2D, get_rotation
FLAGS = flags.FLAGS

import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tools.eval_utils import load_depth, get_bbox
from tools.dataset_utils import *
from evaluation.eval_utils_cass import get_3d_bbox, transform_coordinates_3d, compute_3d_iou_new




class PoseDataset(data.Dataset):
    def __init__(self, source=None, mode='test',
                 n_pts=1024, img_size=256):
        '''
        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        self.source = source
        self.mode = mode
        data_dir = FLAGS.dataset_dir
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size
        self.detection_dir = FLAGS.detection_dir

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']

        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]
        img_list = []
        subset_len = []
        #  aggregate all availabel datasets
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        if source == 'CAMERA':
            self.id2cat_name = self.id2cat_name_CAMERA

        per_obj = FLAGS.per_obj
        self.per_obj = per_obj
        self.per_obj_id = None
        # only test one object
        if self.per_obj in self.cat_names and self.per_obj != 'can' and self.source == 'Real':
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, img_list_cache_filename))]
            else:
                # needs to reorganize img_list
                s_obj_id = self.cat_name2id[self.per_obj]
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gt_path = os.path.join(self.data_dir, img_list[i] + '_label.pkl')
                    try:
                        with open(gt_path, 'rb') as f:
                            gts = cPickle.load(f)
                        id_list = gts['class_ids']
                        if s_obj_id in id_list:
                            img_list_obj.append(img_list[i])
                    except:
                        print(f'WARNING {gt_path} is empty')
                        continue
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
                # iter over  all img_list, cal sublen

            if len(subset_len) == 2:
                camera_len  = 0
                real_len = 0
                for i in range(len(img_list_obj)):
                    if 'CAMERA' in img_list_obj[i].split('/'):
                        camera_len += 1
                    else:
                        real_len += 1
                self.subset_len = [camera_len, real_len]
            #  if use only one dataset
            #  directly load all data
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)

        models = {}
        for path in model_file_path:
            with open(os.path.join(data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))
        self.models = models

        # move the center to the body of the mug
        # meta info for re-label mug category
        with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=np.float)  # [fx, fy, cx, cy]
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)

        self.invaild_list = []

        print('{} images found.'.format(self.length))
        print('{} models loaded.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #   load ground truth
        #  if per_obj is specified, then we only select the target object
        # index = index % self.length  # here something wrong
        img_path = os.path.join(self.data_dir, self.img_list[index])
        if img_path in self.invaild_list:
            return None
        try:
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
        except:
            return None

        if 'CAMERA' in img_path.split('/'):
            out_camK = self.camera_intrinsics
            img_type = 'syn'
        else:
            out_camK = self.real_intrinsics
            img_type = 'real'

        # select one foreground object,
        # if specified, then select the object

        scene = img_path.split('/')[-2]
        img_id = img_path.split('/')[-1]
        if img_type == 'real':
            dataset_split = 'REAL275'
            detection_file = os.path.join(self.detection_dir, dataset_split, f'results_test_{scene}_{img_id}.pkl')
        else:
            dataset_split = 'CAMERA25'
            detection_file = os.path.join(self.detection_dir, dataset_split, f'results_val_{scene}_{img_id}.pkl')
        with open(detection_file, 'rb') as file:
            detection_dict = cPickle.load(file)
        rgb = cv2.imread(img_path + '_color.png')

        if rgb is not None:
            rgb = rgb[:, :, :3]
        else:
            return None

        im_H, im_W = rgb.shape[0], rgb.shape[1]
        depth_path = img_path + '_depth.png'
        if os.path.exists(depth_path):
            depth = load_depth(depth_path)
        else:
            return None
        num_instance = len(detection_dict['pred_class_ids'])

        roi_imgs = []
        roi_depths = []
        roi_masks = []
        roi_depth_norms = []
        sym_infos = []
        mean_shapes = []
        obj_ids = []
        obj_ids_0base = []
        roi_coord_2ds = []
        obj_valid_index = []

        for j in range(num_instance):
            cat_id = detection_dict['pred_class_ids'][j]
            if self.per_obj_id is not None:
                if cat_id != self.per_obj_id:
                    continue
                else:
                    obj_valid_index.append(j)

            coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)
            # aggragate information about the selected object
            mask = detection_dict['pred_masks'][:, :, j]
            bbox = detection_dict['pred_bboxes'][j]
            rmin, rmax, cmin, cmax = get_bbox(bbox)
            # here resize and crop to a fixed size 256 x 256
            bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
            x1, y1, x2, y2 = bbox_xyxy
            # here resize and crop to a fixed size 256 x 256
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bbox_center = np.array([cx, cy])  # (w/2, h/2)
            scale = max(y2 - y1, x2 - x1)
            scale = min(scale, max(im_H, im_W)) * 1.0

            ## roi_image ------------------------------------
            roi_img = crop_resize_by_warp_affine(
                rgb, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1)
            # import matplotlib.pyplot as plt
            # plt.imshow(roi_img.transpose(1, 2, 0))
            # plt.show()
            # roi_coord_2d ----------------------------------------------------
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            ).transpose(2, 0, 1)
            mask_target = mask.copy().astype(np.float)
            # depth[mask_target == 0.0] = 0.0
            roi_mask = crop_resize_by_warp_affine(
                mask_target, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            )
            roi_mask = np.expand_dims(roi_mask, axis=0)
            roi_depth = crop_resize_by_warp_affine(
                depth, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
            )

            roi_depth = np.expand_dims(roi_depth, axis=0)
            # normalize depth
            depth_valid = roi_depth > 0
            if np.sum(depth_valid) <= 1.0:
                return None
            roi_m_d_valid = roi_mask.astype(np.bool) * depth_valid
            if np.sum(roi_m_d_valid) <= 1.0:
                return None

            depth_v_value = roi_depth[roi_m_d_valid]
            depth_normalize = (roi_depth - np.min(depth_v_value)) / (np.max(depth_v_value) - np.min(depth_v_value))
            depth_normalize[~roi_m_d_valid] = 0.0
            # occupancy canonical
            # sym
            sym_info = self.get_sym_info(self.id2cat_name[str(cat_id)])
            mean_shape = self.get_mean_shape(self.id2cat_name[str(cat_id)])
            mean_shape = mean_shape / 1000.0
            roi_imgs.append(roi_img)
            roi_depths.append(roi_depth)
            roi_masks.append(roi_mask)
            roi_depth_norms.append(depth_normalize)
            sym_infos.append(sym_info)
            mean_shapes.append(mean_shape)
            obj_ids.append(cat_id)
            obj_ids_0base.append(cat_id - 1)
            roi_coord_2ds.append(roi_coord_2d)

        if self.per_obj_id is not None:
            for key in ['pred_class_ids', 'pred_bboxes', 'pred_scores']:
                valid_list = []
                for index in obj_valid_index:
                    valid_list.append(detection_dict[key][index])
                detection_dict[key] = np.array(valid_list)
        detection_dict.pop('pred_masks')
        out_camK = np.array([out_camK] * len(roi_imgs))
        roi_imgs = np.array(roi_imgs)
        roi_depths = np.array(roi_depths)
        roi_masks = np.array(roi_masks)
        roi_depth_norms = np.array(roi_depth_norms)
        sym_infos = np.array(sym_infos)
        mean_shapes = np.array(mean_shapes)
        obj_ids = np.array(obj_ids)
        obj_ids_0base = np.array(obj_ids_0base)
        roi_coord_2ds = np.array(roi_coord_2ds)
        data_dict = {}
        data_dict['roi_img'] = torch.as_tensor(roi_imgs.astype(np.float32)).contiguous()
        data_dict['roi_depth'] = torch.as_tensor(roi_depths.astype(np.float32)).contiguous()
        data_dict['cam_K'] = torch.as_tensor(out_camK.astype(np.float32)).contiguous()
        data_dict['roi_mask'] = torch.as_tensor(roi_masks.astype(np.float32)).contiguous()
        data_dict['cat_id'] = torch.as_tensor(obj_ids)
        data_dict['cat_id_0base'] = torch.as_tensor(obj_ids_0base)
        data_dict['depth_normalize'] = torch.as_tensor(roi_depth_norms.astype(np.float32)).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_infos.astype(np.float32)).contiguous()
        data_dict['mean_shape'] = torch.as_tensor(mean_shapes, dtype=torch.float32).contiguous()
        data_dict['roi_coord_2d'] = torch.as_tensor(roi_coord_2ds, dtype=torch.float32).contiguous()
        return data_dict, detection_dict, gts


    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm


    def get_mean_shape(self, c):
        if c == 'bottle':
            unitx = 87
            unity = 220
            unitz = 89
        elif c == 'bowl':
            unitx = 165
            unity = 80
            unitz = 165
        elif c == 'camera':
            unitx = 88
            unity = 128
            unitz = 156
        elif c == 'can':
            unitx = 68
            unity = 146
            unitz = 72
        elif c == 'laptop':
            unitx = 346
            unity = 200
            unitz = 335
        elif c == 'mug':
            unitx = 146
            unity = 83
            unitz = 114
        elif c == '02876657':
            unitx = 324 / 4
            unity = 874 / 4
            unitz = 321 / 4
        elif c == '02880940':
            unitx = 675 / 4
            unity = 271 / 4
            unitz = 675 / 4
        elif c == '02942699':
            unitx = 464 / 4
            unity = 487 / 4
            unitz = 702 / 4
        elif c == '02946921':
            unitx = 450 / 4
            unity = 753 / 4
            unitz = 460 / 4
        elif c == '03642806':
            unitx = 581 / 4
            unity = 445 / 4
            unitz = 672 / 4
        elif c == '03797390':
            unitx = 670 / 4
            unity = 540 / 4
            unitz = 497 / 4
        else:
            unitx = 0
            unity = 0
            unitz = 0
            print('This category is not recorded in my little brain.')
        # scale residual
        return np.array([unitx, unity, unitz])

    def get_origin_scale(self, c, model, nocs_scale):
        # model pc x 3
        lx = max(model[:, 0]) - min(model[:, 0])
        ly = max(model[:, 1]) - min(model[:, 1])
        lz = max(model[:, 2]) - min(model[:, 2])

        # real scale
        lx_t = lx * nocs_scale
        ly_t = ly * nocs_scale
        lz_t = lz * nocs_scale

        return np.array([lx_t, ly_t, lz_t])

    def get_sym_info(self, c):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'camera':
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=np.int)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=np.int)
        elif c == 'mug':
            sym = np.array([0, 1, 0, 0], dtype=np.int)  # for mug, we currently mark it as no symmetry
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        return sym

    def asymmetric_3d_iou(self, RT_1, RT_2, scales_1, scales_2):
        noc_cube_1 = get_3d_bbox(scales_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + \
                np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps
