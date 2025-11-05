/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops
**************************************************************************************************
*/
#include <torch.h>
#include "cpu/ms_deform_attn_cpu.h"

#pragma once

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif


torch::Tensor
ms_deform_attn_forward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int im2col_step)
{
    if (value->is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
      TORCH_ERROR("Not compiled with GPU support");
#endif
    }
    TORCH_ERROR("Not implemented on the CPU");
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
    if (value->is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
      TORCH_ERROR("Not compiled with GPU support");
#endif
    }
    TORCH_ERROR("Not implemented on the CPU");
}
