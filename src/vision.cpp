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
#include "ms_deform_attn.h"

// Wrapper for forward (returns single tensor for R simplicity)
torch::Tensor ms_deform_attn_forward_wrapper(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int8_t im2col_step) {
  auto result = ms_deform_attn_forward(
    value, spatial_shapes, level_start_index,
    sampling_loc, attn_weight, im2col_step);
  return result[0];  // output
}

// Backward wrapper
std::vector<torch::Tensor> ms_deform_attn_backward_wrapper(
    const torch::Tensor &grad_output,
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int8_t im2col_step) {
  auto grad_vector = ms_deform_attn_backward(
    grad_output, value, spatial_shapes,
    level_start_index, sampling_loc, attn_weight, im2col_step);

  return grad_vector;
}

// TORCH_LIBRARY macro: register under a custom namespace
TORCH_LIBRARY(tvision, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward_wrapper);
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward_wrapper);
}
