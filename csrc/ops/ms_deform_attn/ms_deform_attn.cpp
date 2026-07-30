#include "ms_deform_attn.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor ms_deform_attn(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    int64_t im2col_step) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::ms_deform_attn", "")
                       .typed<decltype(ms_deform_attn)>();
  return op.call(
      value,
      spatial_shapes,
      level_start_index,
      sampling_loc,
      attn_weight,
      im2col_step);
}

namespace detail {

std::tuple<at::Tensor, at::Tensor, at::Tensor> _ms_deform_attn_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    int64_t im2col_step) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torchvision::_ms_deform_attn_backward", "")
          .typed<decltype(_ms_deform_attn_backward)>();
  return op.call(
      value,
      spatial_shapes,
      level_start_index,
      sampling_loc,
      attn_weight,
      grad_output,
      im2col_step);
}

} // namespace detail

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::ms_deform_attn(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, int im2col_step) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::_ms_deform_attn_backward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, Tensor grad_output, int im2col_step) -> (Tensor, Tensor, Tensor)"));
}

} // namespace ops
} // namespace vision
