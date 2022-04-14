# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import cv2
import imageio
from PIL import Image
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils


# ## data augmentation: crop, resize, jitter ###
def crop(img, bbox, bgval=0, mode='const'):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.
    Args:
        img: image to crop
        bbox: bounding box to crop: x1, y1, x2, yw
        bgval: default background for regions outside image
    '''
    bbox = [int(round(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    if len(im_shape) < 3:
        img = img[..., None]
    nc = img.shape[2]

    img_out = np.ones((bheight, bwidth, nc)) * bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg

    if mode == 'const':
        img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
    elif mode == 'reflect':
        cropped = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
        img_out = np.pad(cropped, ((y_min_trg, bheight - y_max_trg), (x_min_trg, bwidth - x_max_trg), (0, 0)), 'reflect')

    return img_out


def resize(img, dst_size):
    """Jun-Yan: 'the only correct lib to rescale :p'
    :return: resized numpy.uint8 in shape of (H, W, 3/1), [0, 255]
    """
    if isinstance(dst_size, int):
        dst_size = [dst_size, dst_size]
    if img.shape[-1] == 1:
        # todo: need to convert to 2-dim
        img = img.squeeze(-1)
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize(dst_size, Image.BICUBIC)
    img = np.asarray(img)

    if img.ndim == 2:
        img = img[..., None]
    return img


def color_jitter(im, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    im = im.astype(np.float32) / 255
    f = np.random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)
    f = np.random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    f = np.random.uniform(-hue, hue) * 360.
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + f, 0., 360.)
    f = np.random.uniform(-saturation, saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + f, 0., 1.)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    im = (im * 255).clip(0, 255).astype(np.uint8)
    return im




# ######################## Visualization code ########################

def save_images(images, fname, text_list=[None],col=8, scale=False, bg=None, mask=None, r=0.5,
                save_merge=True, save_idv=False):
    """
    :param it:
    :param images: Tensor of (N, C, H, W)
    :param text_list: str * N
    :param name:
    :param scale: if RGB is in [-1, 1]
    :return:
    """
    def write_to_png(images, png_name, text_list, col):
        merge_image = tensor_text_to_canvas(images, text_list, col=col, scale=scale)
        if png_name is not None:
            os.makedirs(os.path.dirname(png_name), exist_ok=True)
            imageio.imwrite(png_name + '.png', merge_image)
        return merge_image

    if bg is not None:
        images = blend_images(images, bg, mask, r)
    if save_merge:
        merge_image = write_to_png(images, fname, text_list, col=col)
    if save_idv:
        for n in range(len(images)):
            os.makedirs(fname, exist_ok=True)
            write_to_png( images[n: n+1], os.path.join(fname, '%d' % n),[text_list[n]], col=1)

    return merge_image



def save_gifs(image_list, fname, text_list=[None], col=8, scale=False, save_merge=True, save_idv=False):
    """
    :param image_list: [(N, C, H, W), ] * T
    :param fname:
    :return:
    """

    def write_to_gif(gif_name, tensor_list, batch_text=[None], col=8, scale=False):
        """
        :param gif_name: without ext
        :param tensor_list: list of [(N, C, H, W) ] of len T.
        :param batch_text: T * N * str. Put it on top of
        :return:
        """
        T = len(tensor_list)
        if batch_text is None:
            batch_text = [None]
        if len(batch_text) == 1:
            batch_text = batch_text * T
        image_list = []
        for t in range(T):
            time_slices = tensor_text_to_canvas(
                tensor_list[t], batch_text[t], col=col, scale=scale)  # numpy (H, W, C) of uint8
            image_list.append(time_slices)

        if not os.path.exists(os.path.dirname(gif_name)):
            os.makedirs(os.path.dirname(gif_name))
            print('## Make directory: %s' % gif_name)
        imageio.mimsave(gif_name + '.gif', image_list)
        print('save to ', gif_name + '.gif')

    # merge write
    if len(image_list) == 0:
        print('not save empty gif list')
        return
    num = image_list[0].size(0)
    if save_merge:
        write_to_gif(fname, image_list, text_list, col=min(col, num), scale=scale)
    if save_idv:
        for n in range(num):
            os.makedirs(fname, exist_ok=True)
            single_list = [each[n:n+1] for each in image_list]
            write_to_gif(os.path.join(fname, '%d' % n), single_list, [text_list[n]], col=1, scale=scale)



def merge_to_numpy(image_tensor, scale=True, n_col=4, text=None):
    if text is None and (torch.is_tensor(image_tensor) or torch.is_tensor(image_tensor[0])):
        if torch.is_tensor(image_tensor):
            img = image_tensor.cpu().detach()
        else:
            img = [e.cpu().detach() for e in image_tensor]
        img = vutils.make_grid(img, nrow=n_col)
        if scale:
            img = inverse_transform(img)
        img = img.numpy().transpose([1, 2, 0])
        img = np.clip(255 * img, 0, 255).astype(np.uint8)
    else:
        if torch.is_tensor(image_tensor):
            img = image_tensor.cpu().detach().numpy()
        else:
            img = image_tensor
        n_col = min(n_col, img.shape[0])
        n_row = int(img.shape[0] / n_col)
        img = img[0: n_col * n_row]
        img = np.transpose(img, [0, 2, 3, 1])
        img = np.clip(255 * img, 0, 255)
        img = put_text_batch(img, text)
        img = merge(img, [n_row, n_col]).astype(np.uint8)
    return img


def merge(images, size):
    n_row, n_col = size
    n_col = min(n_col, images.shape[0])
    n_row = int(images.shape[0] / n_col)
    N = n_col * n_row
    size = (n_row, n_col)

    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx in range(N):
            image = images[idx]
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1], 1))
        for idx in range(N):
            image = images[idx]
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image[:, :, :]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def blend_images(fg, bg, mask=None, r=0.5):
    if mask is None:
        image = fg * r + bg * (1-r)
    else:
        image = bg * (1 - mask) + (fg * r + bg * (1 - r)) * mask
    return image


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    img = Image.fromarray(x[j:j + crop_h, i:i + crop_w])
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    img = np.asarray(img)
    return img


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True, scale=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = Image.fromarray(image).resize([resize_height, resize_width], Image.BILINEAR)
    if len(cropped_image.shape) != 3:  # In case of binary mask with no channels:
        cropped_image = np.expand_dims(cropped_image, -1)
    if scale:
        cropped_image = np.array(cropped_image)[:, :, :3] / 127.5 - 1.  # [-1, 1]
    cropped_image = cropped_image.transpose([2, 0, 1])
    return cropped_image


def inverse_transform(images):
    images = (images + 1.) / 2.
    # images = images.transpose([1, 2, 0])
    return images

def inv_transform_to_numpy(img, scale=True):
    if torch.is_tensor(img):
        img = img.cpu().detach().numpy()
    if scale:
        img = inverse_transform(img)
    img = np.transpose(img, [0, 2, 3, 1])
    img = np.clip(255 * img, 0, 255)

    img_list = []
    for i in range(img.shape[0]):
        img_list.append(img[i].astype(np.uint8).copy())

    return img_list




# ### print text on images ###
def tensor_text_to_canvas(image, text=None, col=8, scale=False):
    """
    :param image: Tensor / numpy in shape of (N, C, H, W)
    :param text: [str, ] * N
    :param col:
    :return: uint8 numpy of (H, W, C), in scale [0, 255]
    """
    if scale:
        image = image / 2 + 0.5
    if torch.is_tensor(image):
        image = image.cpu().detach().numpy()

    image = write_text_on_image(image, text)  # numpy (N, C, H, W) in scale [0, 1]
    image = vutils.make_grid(torch.from_numpy(image), nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)
    return image


def write_text_on_image(images, text):
    """
    :param images: (N, C, H, W) in scale [0, 1]
    :param text: (str, ) * N
    :return: (N, C, H, W) in scale [0, 1]
    """
    if text is None or text[0] is None:
        return images

    images = np.transpose(images, [0, 2, 3, 1])
    images = np.clip(255 * images, 0, 255).astype(np.uint8)

    image_list = []
    for i in range(images.shape[0]):
        img = images[i].copy()
        img = put_multi_line(img, text[i])
        image_list.append(img)
    image_list = np.array(image_list).astype(np.float32)
    image_list = image_list.transpose([0, 3, 1, 2])
    image_list = image_list / 255
    return image_list


def put_multi_line(img, multi_line, h=15):
    for i, line in enumerate(multi_line.split('\n')):
        img = cv2.putText(img, line, (h, h * (i + 1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    return img


def scatter_sphere(az, el, img_path, color="k"):
    """vis camera pose"""
    # plot sphere
    plt.close()
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    xx = cos(el) * cos(az)
    yy = cos(el) * sin(az)
    zz = sin(el)

    ax.scatter(xx, yy, zz, color=color, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.tight_layout()
    plt.savefig(img_path + '.jpg')

