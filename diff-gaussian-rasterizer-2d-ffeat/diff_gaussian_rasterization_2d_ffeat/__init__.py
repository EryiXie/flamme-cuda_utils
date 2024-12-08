#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple

import torch
import torch.nn as nn

from . import _C_2d
import time


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    sh,
    colors_precomp,
    ffeats,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta, # Camera pose rotation
    rho, # Camera pose translation
    tile_mask,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        sh,
        colors_precomp,
        ffeats,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta, # Camera pose rotation
        rho, # Camera pose translation
        tile_mask,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D, 
        sh,
        colors_precomp,
        ffeats, # Added foundation features
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta, # Camera pose rotation
        rho, # Camera pose translation
        tile_mask,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            ffeats, # Added foundation features
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw, # Used in backward propagation
            tile_mask,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.cx,
            raster_settings.cy,
            sh,
            raster_settings.sh_degree,
            raster_settings.color_sigma,
            raster_settings.campos,
            raster_settings.opaque_threshold,
            raster_settings.depth_threshold,
            raster_settings.normal_threshold,
            raster_settings.T_threshold,
            raster_settings.prefiltered,
            raster_settings.debug,
        )
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    num_tile,
                    color,
                    depth,
                    hit_color,
                    hit_depth,
                    hit_color_weight,
                    hit_depth_weight,
                    ffeat_map, # Added foundation features
                    T_map,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    n_touched, # TODO: opacity is not used in the forward pass, what's the impact of this?
                    tile_indices,
                ) = _C_2d.rasterize_gaussians(*args) 
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                num_tile,
                color,
                depth,
                hit_color,
                hit_depth,
                hit_color_weight,
                hit_depth_weight,
                ffeat_map, # Added foundation features
                T_map,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                n_touched,
                tile_indices,
            ) = _C_2d.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_tile = num_tile
        ctx.save_for_backward(
            colors_precomp,
            ffeats, # Added foundation features
            hit_depth,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            tile_indices,
        )
        return (
            color,
            depth,
            hit_color,
            hit_depth,
            hit_color_weight,
            hit_depth_weight,
            T_map,
            radii,
            n_touched, # TODO: opacity is not used in the forward pass, what's the impact of this?
            ffeat_map, # Added foundation features
        )

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_out_depth,
        grad_hit_color,
        grad_hit_depth,
        grad_hit_color_weight,
        grad_hit_depth_weight,
        grad_T_map,
        _,
        grad_n_touched,
        grad_out_ffeat_map, # Added gradient for foundation features
    ):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_tile = ctx.num_tile
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            ffeats, # Added foundation features
            hit_depth,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            tile_indices,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            tile_indices,
            num_tile,
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            ffeats, # Added foundation features
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.depth_threshold,
            raster_settings.normal_threshold,
            grad_out_color,
            grad_out_ffeat_map, # Added gradient for foundation features
            grad_out_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            hit_depth,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_ffeats, # Added gradient for foundation features
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    grad_tau # Added gradient for camera pose
                ) = _C_2d.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_ffeats, # Added gradient for foundation features
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
                grad_tau # Added gradient for camera pose
            ) = _C_2d.rasterize_gaussians_backward(*args) # TODO: check the order of outputs
        
        # From gradients tau to gradients theta and rho
        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

        grads = (
            grad_means3D,
            grad_sh,
            grad_colors_precomp,
            grad_ffeats, # Added gradient for foundation features
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta, # Added gradient for camera pose
            grad_rho, # Added gradient for camera pose
            None,
            None,
        ) #TODO: None, None for tile_mask and raster_settings, grads order is the samen as forward input order  @staticmethod def forward(...)
        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    projmatrix_raw : torch.Tensor # Used in backward propagation.cu
    sh_degree: int
    campos: torch.Tensor
    opaque_threshold: float
    normal_threshold: float
    depth_threshold: float
    prefiltered: bool
    debug: bool
    cx: float
    cy: float
    color_sigma: float = 3.0
    T_threshold: float = 0.0001


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C_2d.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        opacities,
        shs=None,
        ffeats=None, # Added foundation features
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        theta=None, # Camera pose rotation
        rho=None, # Camera pose translation
        tile_mask=None,
        normal_w=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])
        # Added foundation features 
        if ffeats is None:
            ffeats = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            shs,
            colors_precomp,
            ffeats, # Added foundation features 
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            theta, # Camera pose rotation
            rho, # Camera pose translation
            tile_mask,
            raster_settings,
        )
