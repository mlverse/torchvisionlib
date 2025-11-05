/*!
 **************************************************************************************************
 * Deformable DETR
 * Copyright (c) 2020 SenseTime. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 **************************************************************************************************
 * Modified from https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops
 **************************************************************************************************
 */
#include <torch/torch.h>
#include <stdexcept>
#include <vector>
#include "cpu/ms_deform_attn_cpu.h"

#pragma once

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

std::vector<torch::Tensor>
  ms_deform_attn_forward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int im2col_step);

std::vector<torch::Tensor>
  ms_deform_attn_backward(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const torch::Tensor &grad_output,
    const int im2col_step);
