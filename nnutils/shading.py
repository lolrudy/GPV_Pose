# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import convert_to_tensors_and_broadcast
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
from pytorch3d.structures import Meshes

"""My feature shader. Inspried by https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/renderer/mesh/shader.py#L154 """

class SoftFeatureGouraudShader(nn.Module):
    """
    Per vertex feature blending - no lighting model is applied. Vertex color (C=3) is generalized to feature C,
    which are then interpolated using the barycentric coordinates to
    obtain the "colors" for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    Here we don't interpolate the normal
    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftFeatureShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self,
        blend_params=None, **kwargs,
    ):
        super().__init__()
        self.blend_params = (
            blend_params if blend_params is not None else BlendParams()
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        feature = kwargs.get("feature")
        lighting = kwargs.get("lighting")
        if not lighting:
            pixel_colors = feature_shading(
                meshes=meshes,
                fragments=fragments,
                feature=feature,
            )
        else:
            raise NotImplementedError
            # feature_light_shading()
        images = softmax_feat_blend(pixel_colors, fragments, self.blend_params, **kwargs)
        return images


def feature_shading(
    meshes: Meshes, fragments, feature
) -> torch.Tensor:
    """
    Apply per vertex shading - just copy the feature. Here no illuminatino is considerd.
    Then interpolate the vertex shaded feature using the barycentric coordinates
    to get a feature per pixel.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        feature: (sum(V), C) Packed!!

    Returns:
        colors: (N, H, W, K, 3)
    """
    faces = meshes.faces_packed()  # (F, 3)

    face_colors = feature[faces]
    colors = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, face_colors
    )
    return colors


def softmax_feat_blend(
    colors, fragments, blend_params, **kwargs
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        colors: (N, H, W, K, C) RGB color for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, C+1) A range from [0, 1]

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    z_plane = kwargs.get('z_plane', 50)
    z_range = kwargs.get('z_range', 50)
    zfar = z_plane + z_range
    znear = z_plane - z_range
    if torch.is_tensor(zfar):
        zfar = zfar.view(zfar.size(0), 1, 1, 1)
        znear = znear.view(znear.size(0), 1, 1, 1)

    N, H, W, K = fragments.pix_to_face.shape
    C = colors.size(-1)
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones(
        (N, H, W, C+1), dtype=colors.dtype, device=colors.device
    )
    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.zeros([C, ], dtype=torch.float32, device=device) + background
        # background = torch.tensor(
        #     background, dtype=torch.float32, device=device
        # )

    # Background color
    delta = np.exp(1e-10 / blend_params.gamma) * 1e-10
    delta = torch.tensor(delta, device=device)

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None]
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta
    weights = weights_num / denom

    # Sum: weights * textures + background color
    weighted_colors = (weights[..., None] * colors).sum(dim=-2)
    weighted_background = (delta / denom) * background
    pixel_colors[..., :C] = weighted_colors + weighted_background
    pixel_colors[..., C] = 1.0 - alpha

    return pixel_colors
