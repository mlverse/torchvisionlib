#include "roi_align_rotated.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor roi_align_rotated(
    const at::Tensor& input,
    const at::Tensor& rois,
    int64_t pooled_height,
    int64_t pooled_width,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    bool clockwise) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_align_rotated", "")
                       .typed<decltype(roi_align_rotated)>();
  return op.call(
      input,
      rois,
      pooled_height,
      pooled_width,
      spatial_scale,
      sampling_ratio,
      aligned,
      clockwise);
}

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
    int64_t width) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_roi_align_rotated_backward", "")
          .typed<decltype(_roi_align_rotated_backward)>();
  return op.call(
      grad_output,
      rois,
      pooled_height,
      pooled_width,
      spatial_scale,
      sampling_ratio,
      aligned,
      clockwise,
      batch_size,
      channels,
      height,
      width);
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::roi_align_rotated(Tensor input, Tensor rois, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned, bool clockwise) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_roi_align_rotated_backward(Tensor grad_output, Tensor rois, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned, bool clockwise, int batch_size, int channels, int height, int width) -> Tensor"));
}

} // namespace ops
} // namespace vision
