/*!
 **************************************************************************************************
 * Deformable DETR
 * Copyright (c) 2020 SenseTime. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 **************************************************************************************************
 * Modified from https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops
 **************************************************************************************************
 */

#include "ms_deform_attn.h"

std::vector<torch::Tensor>
  ms_deform_attn_forward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int im2col_step)
  {
    if (value.is_cuda())
    {
#ifdef WITH_CUDA
      return ms_deform_attn_cuda_forward(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
      throw std::runtime_error("Not compiled with GPU support");
#endif
    }
    return ms_deform_attn_cpu_forward(
      value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
  }

std::vector<torch::Tensor>
  ms_deform_attn_backward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const torch::Tensor &grad_output,
    const int im2col_step)
  {
    if (value.is_cuda())
    {
#ifdef WITH_CUDA
      return ms_deform_attn_cuda_backward(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
      throw std::runtime_error("Not compiled with GPU support");
#endif
    }
    return ms_deform_attn_cpu_backward(
      value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
  }
