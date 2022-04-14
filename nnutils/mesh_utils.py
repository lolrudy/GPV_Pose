# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pytorch3d.ops as ops_3d
from pytorch3d.renderer import MeshRasterizer, SfMPerspectiveCameras, TexturesVertex, MeshRenderer, SoftGouraudShader, \
    DirectionalLights, RasterizationSettings, get_world_to_view_transform
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.utils import _interpolate_zbuf, _clip_barycentric_coordinates

from pytorch3d.structures import Meshes
import pytorch3d.structures.utils as struct_utils
from pytorch3d.transforms import Transform3d, Rotate

from nnutils import geom_utils


def cubify(vox_world, th=0.1, detach_vox=True) -> Meshes:
    """
    scale range from -0.5, 0.5
    :param vox_world: （N， C， D, H, W)
    :param th:
    :return:
    """
    if not torch.is_tensor(vox_world):
        W = vox_world.shape[-1]; N = vox_world.shape[0]
        vox_world = torch.FloatTensor(vox_world).view(N, 1, W, W, W).cuda()
    if detach_vox:
        vox_world = vox_world.detach()
    meshes = ops_3d.cubify(vox_world.squeeze(1), th, align='corner')
    meshes = meshes.scale_verts_(0.5)
    return meshes

def param_to_7dof_batcch(param, f=375, use_scale=False, use_rho=False):
    """
    :param param: (N, 6)
    :param f: scaler
    :param use_scale:
    :param use_rho:
    :return:
    """
    N, C = param.size()
    azel, scale, trans = torch.split(param, [2, 1, 3], dim=1)
    zeros = torch.zeros_like(scale)
    if not use_scale:
        scale = zeros + 1
    if not use_rho:
        trans = torch.cat([zeros, zeros, zeros + calc_rho(f)], dim=1)
    f = zeros + f
    new_param = torch.cat([azel, scale, trans, f], dim=1)
    return new_param


def calc_rho(f):
    base_f = 1.875
    base_rho = 2
    rho = base_rho * f / base_f
    return rho


def view_vox2mesh_py3d(view):
    """
    :param view: (N, 7)
    :return: (N, ), (N, 3, 3), (N, 3)
    """
    view = view.clone()
    view, f = torch.split(view, [6, 1], dim=1)
    view[:, 0] = -view[:, 0]
    f = (f * 2).squeeze(1)

    scale, trans, rot = geom_utils.azel2uni(view, homo=False)
    return f, rot, trans


def param7dof_to_camera(view_params) -> SfMPerspectiveCameras:
    """
    :param view_params: (N, 7)
    :return: SfM cameras
    """
    f, rot, trans = view_vox2mesh_py3d(view_params)
    cameras = SfMPerspectiveCameras(focal_length=f, R=rot, T=trans, device=view_params.device)
    return cameras


def render_meshify_voxel(voxels, out_size, view_param, th=0.05):
    meshes = cubify(voxels, th)
    try:
        recon = render_mesh(meshes, out_size, view_param)
    except:
        print('No mesh')
        N = voxels.size(0)
        recon = {'image': torch.zeros(N, 3, out_size, out_size)}
    return recon


def render_mesh(meshes: Meshes, out_size, view_param, texture=None, **kwargs):
    N, V, _ = meshes.verts_padded().size()
    if meshes.textures is None:
        if texture is None:
            texture = torch.zeros([N, V, 3]).to(view_param) + 1 # torch.FloatTensor([[[175., 175., 175.]]]).to(view_param) / 255
        meshes.textures = pad_texture(meshes, texture)
    cameras = param7dof_to_camera(view_param)

    raster_settings = kwargs.get('raster_settings', RasterizationSettings(image_size=out_size))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    shader = SoftGouraudShader(device=meshes.device, lights=ambient_light(meshes.device, view_param))

    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )
    if 'zfar' not in kwargs:
        kwargs['zfar']= view_param[:, -2].view(N, 1, 1, 1) + 1
    if 'znear' not in kwargs:
        kwargs['znear'] = view_param[:, -2].view(N, 1, 1, 1) - 1

    image = renderer(meshes, cameras=cameras,  **kwargs)

    image = torch.flip(image, dims=[-3])
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]

    image = image * 2 - 1
    # rgb, mask = torch.split(output, [3, 1], dim=1)
    return {'image': rgb, 'mask': mask, 'rgba': image}


def render_normals(meshes: Meshes, out_size, view_param, **kwargs):
    N, V, _ = meshes.verts_padded().size()
    # clone mesh to and replace texture with normals in camera space
    meshes = meshes.clone()
    world_normals = meshes.verts_normals_padded()
    cameras = param7dof_to_camera(view_param)  # real camera
    trans_world_to_view = cameras.get_world_to_view_transform()
    view_normals = trans_world_to_view.transform_normals(world_normals)

    # place view normal as textures
    meshes.textures = pad_texture(meshes, view_normals)

    raster_settings = kwargs.get('raster_settings', RasterizationSettings(image_size=out_size))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    # make the ambient color full range
    shader = SoftGouraudShader(device=meshes.device, lights=ambient_light(meshes.device, view_param, color=(1, 0, 0)))

    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )
    if 'zfar' not in kwargs:
        kwargs['zfar']= view_param[:, -2].view(N, 1, 1, 1) + 1
    if 'znear' not in kwargs:
        kwargs['znear'] = view_param[:, -2].view(N, 1, 1, 1) - 1

    image = renderer(meshes, cameras=cameras,  **kwargs)

    image = torch.flip(image, dims=[-3])
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]

    # align w/ my def
    # flip r (x), b (z)
    rgb[:, 0] *= -1
    rgb[:, 2] *= -1
    # mask out bg
    rgb = rgb * mask
    # and normalize rgb to unit vector.
    rgb = F.normalize(rgb, dim=1)  # N, 3, H, W

    # rgb, mask = torch.split(output, [3, 1], dim=1)
    return {'normal': rgb, 'mask': mask, 'rgba': image}


def pad_texture(meshes: Meshes, feature: torch.Tensor) -> TexturesVertex:
    """
    :param meshes:
    :param feature: (sumV, C)
    :return:
    """
    if isinstance(feature, TexturesVertex):
        return feature
    if feature.dim() == 2:
        feature = struct_utils.packed_to_list(feature, meshes.num_verts_per_mesh().tolist())
        # feature = struct_utils.list_to_padded(feature, pad_value=-1)

    texture = TexturesVertex(feature)
    texture._num_faces_per_mesh = meshes.num_faces_per_mesh().tolist()
    texture._num_verts_per_mesh = meshes.num_verts_per_mesh().tolist()
    texture._N = meshes._N
    texture.valid = meshes.valid
    return texture



def ambient_light(device='cpu', param_view=None, **kwargs):
    amb = 0.6
    if param_view is None:
        d = get_light_direction(param_view)
    else:
        d = ((0, -0.6, 0.8), )

    color = kwargs.get('color', np.array([0.65, 0.3, 0.0]))
    am, df, sp = color
    ambient_color=((am, am, am), ),
    diffuse_color=((df, df, df),),
    specular_color=((sp, sp, sp), ),

    lights = DirectionalLights(
        device=device,
        ambient_color=ambient_color,
        diffuse_color=diffuse_color,
        specular_color=specular_color,
        direction=d,
    )
    return lights


def get_light_direction(view_params):
    """same el, opposite az"""
    N = view_params.size(0)
    az, el, _ = torch.split(view_params, [1, 1, view_params.size(-1) - 2], dim=-1)
    az = -az # np.pi

    rot = geom_utils.azel2rot(az, el, False)  # (N, 3, 3)
    unit = torch.zeros([N, 3, 1]).to(az)
    unit[:, 2] += 1  # z += 1
    unit = torch.matmul(rot, unit).squeeze(-1)
    return -unit


def get_soft_rasterizer_setting(**kwargs):
    sigma = kwargs.get('sigma', 1e-4)
    raster_settings_soft = RasterizationSettings(
        image_size=kwargs.get('image_size', 224),
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=kwargs.get('faces_per_pixel', 10),
        perspective_correct=False,
    )
    return raster_settings_soft



def get_local_feat(meshes: Meshes, rasterizer: MeshRasterizer, local_feat, view, sym=1, vis=True
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    _, screen_meshes, img_vis = transform_to_screen_vis(meshes, rasterizer, view, True)
    verts_feat = sample_local_feat(local_feat, screen_meshes, img_vis, vis=vis)

    if sym > 0:
        sym_index = find_sym_index(meshes, dim=0)
        verts_feat = symmetrify_verts_feat(verts_feat, img_vis, sym_index, mode=sym)
        img_vis = symmetrify_verts_feat(img_vis, img_vis, sym_index, mode=1)
    return verts_feat, img_vis


def transform_to_screen_vis(meshes: Meshes, rasterizer: MeshRasterizer, view, vis=True):
    fragments, screen_meshes = rasterize_wrapper(rasterizer, meshes, view)
    img_vis = get_vis_verts(meshes, fragments, visible=vis)
    return fragments, screen_meshes, img_vis


def get_vis_verts(world_meshes: Meshes, fragment: Fragments, visible=True,
                  ) -> torch.Tensor:
    device = world_meshes.device
    if visible:
        # consider visible
        face_inds = fragment.pix_to_face[...,0]  # (N, H, W, K) LongTensor in packed faces
        valid_inds = face_inds > -1
        face_valid_inds = torch.masked_select(face_inds, valid_inds)  # nFvalid
        world_meshes.faces_padded()
        faces = world_meshes.faces_packed()  # (nF, 3)
        vis_verts_inds = faces[face_valid_inds, :].view(-1) # (nFvalid * 3)

        visible_verts = torch.zeros([world_meshes.verts_packed().size(0)]).to(vis_verts_inds) # (sumV, )
        visible_verts.scatter_(0, vis_verts_inds, 1.)
        visible_verts = visible_verts.unsqueeze(-1)
    else:
        visible_verts = torch.ones([world_meshes.verts_packed().size(0), 1]).to(device) # (sumV, )

    return visible_verts


def symmetrify_verts_feat(verts_feat, wgt, sym_index, dim=0, mode=1, eps=1e-6) -> torch.Tensor:
    """
    :param verts_feat: packed feature (sumV, C)
    :param wgt: packed wgt (sumV, C)
    :param sym_index: meshes or symmetric index.
    :return:
    """
    if isinstance(sym_index, Meshes):
        sym_index = find_sym_index(sym_index, dim)

    flip_feat = torch.gather(verts_feat, 0, sym_index.expand(verts_feat.size()))
    flip_w = torch.gather(wgt, 0, sym_index)

    if mode == 1:
        # all avg.
        verts_feat = (verts_feat * wgt + flip_feat * flip_w) / (flip_w + wgt + eps)
    elif mode == 2:
        # 1,1. 1,0, 0,0 -> myself. 0,1 -> symmetry
        mask = (1-wgt) * flip_w
        verts_feat = (1-mask) * verts_feat + mask * flip_feat

    return verts_feat


def find_sym_index(world_meshes, dim=0):
    world_verts = world_meshes.verts_padded()
    flip_world_verts = world_verts.clone()
    flip_world_verts[..., dim] = -flip_world_verts[..., dim]
    l1 = world_meshes.num_verts_per_mesh()
    _, p1_index, _ = ops_3d.knn_points(world_verts, flip_world_verts, l1, l1)
    # convert from padded to packed.
    offset = world_meshes.mesh_to_verts_packed_first_idx()  # (N, )
    p1_index = p1_index.squeeze(-1) + offset.unsqueeze(-1)  # (N, maxV)
    p1_index = p1_index.view(-1, 1)
    packed_index = world_meshes.verts_padded_to_packed_idx().unsqueeze(-1)  # 1D. (sumV, 1)
    p1_packed_index = torch.gather(p1_index, 0, packed_index)  # (sumV, 1)

    return p1_packed_index


def rasterize_wrapper(rasterizer: MeshRasterizer, meshes: Meshes, param_view: torch.Tensor) -> (Fragments, Meshes):
    """
    from pytroch3d, fix bugs in z_buffer
    :param rasterizer:
    :param meshes: Meshes in world coordinate
    :param param_view: (N, 7)
    :return: Fragment, and Meshes in screen space. Z is in camera space for now.
    """
    cameras = param7dof_to_camera(param_view)
    fragments = rasterizer(meshes, cameras=cameras)

    # cmt: no idea what these do. copy from pytorch3d/mesh/renderer.py:forward
    raster_settings = rasterizer.raster_settings
    if raster_settings.blur_radius > 0.0:
        print('no')
        # TODO: potentially move barycentric clipping to the rasterizer
        # if no downstream functions requires unclipped values.
        # This will avoid unnecssary re-interpolation of the z buffer.
        meshes_screen = rasterizer.transform(meshes, cameras=cameras)
        clipped_bary_coords = _clip_barycentric_coordinates(
            fragments.bary_coords
        )
        clipped_zbuf = _interpolate_zbuf(
            fragments.pix_to_face, clipped_bary_coords, meshes_screen
        )
        fragments = Fragments(
            bary_coords=clipped_bary_coords,
            zbuf=clipped_zbuf,
            dists=fragments.dists,
            pix_to_face=fragments.pix_to_face,
        )

    meshes_screen = transform_verts(meshes, cameras=cameras)
    return fragments, meshes_screen


def sample_local_feat(img_feats, screen_meshes: Meshes, visible_verts,
                       interp_mode: str = "bilinear",
                       padding_mode: str = "zeros",
                       align_corners: bool = False,
                       vis: bool = True,
                       bg_value=0.) -> Tuple[torch.Tensor, torch.Tensor]:
    """if vis: invisible verts are getting bg_value. """
    verts_feat = ops_3d.vert_align(img_feats, screen_meshes,
                                   return_packed=True, interp_mode=interp_mode,
                                   padding_mode=padding_mode, align_corners=align_corners)  # (sumV, C)
    # consider visible
    if vis:
        verts_feat = verts_feat * visible_verts + (1 - visible_verts) * bg_value # (sumV, 1)
    return verts_feat



def transform_verts(meshes_world: Meshes, cameras, flip_x=True, **kwargs) -> Meshes:
    """
    copy from rasterizer.transform()
    Args:
        meshes_world: a Meshes object representing a batch of meshes with
            vertex coordinates in world space.

    Returns:
        meshes_screen: a Meshes object with the vertex positions in screen
        space

    NOTE: keeping this as a separate function for readability but it could
    be moved into forward.
    """
    verts_world = meshes_world.verts_padded()
    verts_world_packed = meshes_world.verts_packed()
    verts_screen = cameras.transform_points(verts_world, **kwargs)

    # NOTE: Retaining view space z coordinate for now.
    # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
    # [0, 1] range.
    view_transform = get_world_to_view_transform(R=cameras.R, T=cameras.T)
    verts_view = view_transform.transform_points(verts_world)
    verts_screen[..., 2] = verts_view[..., 2]
    # cmt: why do we need to flip x-axis instead of y???
    if flip_x:
        verts_screen[..., 0] = -verts_screen[..., 0]

    # Offset verts of input mesh to reuse cached padded/packed calculations.
    pad_to_packed_idx = meshes_world.verts_padded_to_packed_idx()
    verts_screen_packed = verts_screen.view(-1, 3)[pad_to_packed_idx, :]
    verts_packed_offset = verts_screen_packed - verts_world_packed
    return meshes_world.offset_verts(verts_packed_offset)


def transform_meshes(geom: Meshes, trans: Transform3d):
    if isinstance(trans, torch.Tensor):
        camera = param7dof_to_camera(trans)
        trans = Rotate(camera.R, device=camera.device)
    new_verts = trans.transform_points(geom.verts_padded())
    return geom.update_padded(new_verts)
