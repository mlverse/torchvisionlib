#include "../ms_deform_attn.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

class MSDeformAttnFunction
    : public torch::autograd::Function<MSDeformAttnFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& value,
      const torch::autograd::Variable& spatial_shapes,
      const torch::autograd::Variable& level_start_index,
      const torch::autograd::Variable& sampling_loc,
      const torch::autograd::Variable& attn_weight,
      int64_t im2col_step) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto output = ms_deform_attn(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        im2col_step);

    ctx->save_for_backward(
        {value, spatial_shapes, level_start_index, sampling_loc, attn_weight});
    ctx->saved_data["im2col_step"] = im2col_step;

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    auto saved = ctx->get_saved_variables();
    auto value = saved[0];
    auto spatial_shapes = saved[1];
    auto level_start_index = saved[2];
    auto sampling_loc = saved[3];
    auto attn_weight = saved[4];

    auto im2col_step = ctx->saved_data["im2col_step"].toInt();

    auto grads = detail::_ms_deform_attn_backward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        grad_output[0],
        im2col_step);

    auto grad_value = std::get<0>(grads);
    auto grad_sampling_loc = std::get<1>(grads);
    auto grad_attn_weight = std::get<2>(grads);

    return {
        grad_value,
        torch::autograd::Variable(), // spatial_shapes
        torch::autograd::Variable(), // level_start_index
        grad_sampling_loc,
        grad_attn_weight,
        torch::autograd::Variable(), // im2col_step
    };
  }
};

at::Tensor ms_deform_attn_autograd(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    int64_t im2col_step) {
  return MSDeformAttnFunction::apply(
      value,
      spatial_shapes,
      level_start_index,
      sampling_loc,
      attn_weight,
      im2col_step)[0];
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::ms_deform_attn"),
      TORCH_FN(ms_deform_attn_autograd));
}

} // namespace ops
} // namespace vision
