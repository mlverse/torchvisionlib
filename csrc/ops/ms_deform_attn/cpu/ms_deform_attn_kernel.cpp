#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

// No CPU kernel: ms_deform_attn is CUDA-only. On CPU, models fall back to a
// grid_sample-based implementation.
at::Tensor ms_deform_attn_forward_kernel(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    int64_t im2col_step) {
  TORCH_CHECK(false, "ms_deform_attn is not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ms_deform_attn_backward_kernel(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    int64_t im2col_step) {
  TORCH_CHECK(false, "ms_deform_attn is not implemented on the CPU");
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::ms_deform_attn"),
      TORCH_FN(ms_deform_attn_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_ms_deform_attn_backward"),
      TORCH_FN(ms_deform_attn_backward_kernel));
}

} // namespace ops
} // namespace vision
