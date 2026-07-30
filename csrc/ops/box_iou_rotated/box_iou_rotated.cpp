#include "box_iou_rotated.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::box_iou_rotated", "")
                       .typed<decltype(box_iou_rotated)>();
  return op.call(boxes1, boxes2);
}

// Vendored because the pinned TorchVision (v0.23.0) has no box_iou_rotated.
// Drop this directory if TorchVision is bumped to a release that ships the op,
// otherwise the schema below is registered twice and loading fails.
TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::box_iou_rotated(Tensor boxes1, Tensor boxes2) -> Tensor"));
}

} // namespace ops
} // namespace vision
