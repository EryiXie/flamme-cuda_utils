/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({(long long)N});
		return reinterpret_cast<char *>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const torch::Tensor& projmatrix_raw,
	const torch::Tensor &render_mask,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const float cx,
	const float cy,
	const torch::Tensor &sh,
	const int degree,
	const float color_sigma,
	const torch::Tensor &campos,
	const float opaque_threshold,
	const float hit_depth_threshold,
	const float hit_normal_threshold,
	const float T_threshold,
	const bool prefiltered,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor out_hit_depth = torch::full({1, H, W}, 0, int_opts);
	torch::Tensor out_hit_color = torch::full({1, H, W}, 0, int_opts);
	torch::Tensor tile_indices = torch::full({H * W}, -1, int_opts);
	torch::Tensor out_T = torch::full({1, H, W}, 1, float_opts);
	torch::Tensor out_hit_color_weight = torch::full({1, H, W}, 0, float_opts);
	torch::Tensor out_hit_depth_weight = torch::full({1, H, W}, 0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor n_touched = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int tile_num = 0;

	int rendered = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			color_sigma,
			background.contiguous().data<float>(),
			W, H,
			cx, cy,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(),
			opacity.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			render_mask.contiguous().data<int>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			opaque_threshold,
			hit_depth_threshold,
			hit_normal_threshold,
			T_threshold,
			tile_indices.contiguous().data<int>(),
			out_color.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			out_hit_depth.contiguous().data<int>(),
			out_hit_color.contiguous().data<int>(),
			out_hit_color_weight.contiguous().data<float>(),
			out_hit_depth_weight.contiguous().data<float>(),
			out_T.contiguous().data<float>(),
			tile_num,
			radii.contiguous().data<int>(),
			n_touched.contiguous().data<int>(), //TODO: add this also to forward.cu 
			debug);
	}
	return std::make_tuple(rendered, tile_num, out_color, out_depth,
						   out_hit_color, out_hit_depth, out_hit_color_weight, out_hit_depth_weight,
						   out_T, radii, geomBuffer, binningBuffer, imgBuffer, n_touched, tile_indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &tile_indices,
	const int tile_num,
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &colors,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const torch::Tensor& projmatrix_raw,
	const float tan_fovx,
	const float tan_fovy,
	const float cx, const float cy,
	const float depth_threshold,
	const float normal_threshold,
	const torch::Tensor &dL_dout_color,
	const torch::Tensor &dL_dout_depth,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const torch::Tensor &hit_image,
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0)
	{
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dtau = torch::zeros({P,6}, means3D.options());

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(P, degree, M, R,
											 tile_num,
											 tile_indices.contiguous().data<int>(),
											 background.contiguous().data<float>(),
											 W, H,
											 cx, cy,
											 means3D.contiguous().data<float>(),
											 sh.contiguous().data<float>(),
											 colors.contiguous().data<float>(),
											 scales.data_ptr<float>(),
											 scale_modifier,
											 rotations.data_ptr<float>(),
											 cov3D_precomp.contiguous().data<float>(),
											 viewmatrix.contiguous().data<float>(),
											 projmatrix.contiguous().data<float>(),
											 projmatrix_raw.contiguous().data<float>(),
											 campos.contiguous().data<float>(),
											 tan_fovx,
											 tan_fovy,
											 normal_threshold,
											 depth_threshold,
											 radii.contiguous().data<int>(),
											 reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
											 dL_dout_color.contiguous().data<float>(),
											 dL_dout_depth.contiguous().data<float>(),
											 hit_image.contiguous().data<int>(),
											 dL_dmeans2D.contiguous().data<float>(),
											 dL_dconic.contiguous().data<float>(),
											 dL_dopacity.contiguous().data<float>(),
											 dL_dcolors.contiguous().data<float>(),
											 dL_dmeans3D.contiguous().data<float>(),
											 dL_dcov3D.contiguous().data<float>(),
											 dL_dsh.contiguous().data<float>(),
											 dL_dscales.contiguous().data<float>(),
											 dL_drotations.contiguous().data<float>(),
											 dL_dtau.contiguous().data<float>(), //TODO: add this also to backward.cu
											 debug);
	}
	return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dtau);
}

torch::Tensor markVisible(
	torch::Tensor &means3D,
	torch::Tensor &viewmatrix,
	torch::Tensor &projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(P,
												means3D.contiguous().data<float>(),
												viewmatrix.contiguous().data<float>(),
												projmatrix.contiguous().data<float>(),
												present.contiguous().data<bool>());
	}

	return present;
}