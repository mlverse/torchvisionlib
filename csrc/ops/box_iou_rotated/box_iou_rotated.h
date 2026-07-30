#pragma once

#include <ATen/ATen.h>

namespace vision {
namespace ops {

at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);

} // namespace ops
} // namespace vision
