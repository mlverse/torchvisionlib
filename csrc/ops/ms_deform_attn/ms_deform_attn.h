#pragma once

#include <ATen/ATen.h>

namespace vision {
namespace ops {

// Multi-scale deformable attention, as used by Deformable-DETR / LW-DETR.
// Only a CUDA kernel is provided; calling on CPU tensors raises an error.
at::Tensor ms_deform_attn(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    int64_t im2col_step);

namespace detail {

std::tuple<at::Tensor, at::Tensor, at::Tensor> _ms_deform_attn_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    int64_t im2col_step);

} // namespace detail

} // namespace ops
} // namespace vision
