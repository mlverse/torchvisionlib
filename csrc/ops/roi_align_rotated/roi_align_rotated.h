#pragma once

#include <ATen/ATen.h>

namespace vision {
namespace ops {

// RoI align pooling for rotated proposals, matching the MMCV
// `roi_align_rotated` operator:
// https://mmcv.readthedocs.io/en/latest/deployment/mmcv_ops_definition.html#mmcvroialignrotated
//
// rois is a `Tensor[K, 6]` with columns `(batch_index, cx, cy, w, h, angle)`
// where `batch_index` is 0-based, the box is centered at `(cx, cy)` with size
// `(w, h)` and `angle` is expressed in radians.
at::Tensor roi_align_rotated(
    const at::Tensor& input,
    const at::Tensor& rois,
    int64_t pooled_height,
    int64_t pooled_width,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    bool clockwise);

namespace detail {

at::Tensor _roi_align_rotated_backward(
    const at::Tensor& grad_output,
    const at::Tensor& rois,
    int64_t pooled_height,
    int64_t pooled_width,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    bool clockwise,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);

} // namespace detail

} // namespace ops
} // namespace vision
