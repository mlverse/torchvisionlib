
#include "ms_deform_attn_autograd.h"
#include "ms_deform_attn.h"

// Autograd function
torch::Tensor MSDeformAttnFunction::forward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int im2col_step
) {
  ctx->save_for_backward(
      {value, spatial_shapes, level_start_index, sampling_loc, attn_weight}
  );
  ctx->saved_data["im2col_step"] = im2col_step;

  torch::Tensor out = torchvision::ms_deform_attn_forward(
    value,
    spatial_shapes,
    level_start_index,
    sampling_loc,
    attn_weight,
    im2col_step);

  return out;
  }

torch::autograd::tensor_list MSDeformAttnFunction::backward(
    torch::autograd::AutogradContext *ctx,
    const torch::Tensor &grad_output)
  {
  auto saved = ctx->get_saved_variables();
  const torch::Tensor &value            = saved[0];
  const torch::Tensor &spatial_shapes   = saved[1];
  const torch::Tensor &level_start_idx  = saved[2];
  const torch::Tensor &sampling_loc     = saved[3];
  const torch::Tensor &attn_weight      = saved[4];
  const int im2col_step = ctx->saved_data["im2col_step"].toInt();

  std::vector<torch::Tensor> grads =
    torchvision::ms_deform_attn_backward(
      grad_output,
      value,
      spatial_shapes,
      level_start_idx,
      sampling_loc,
      attn_weight,
      im2col_step);

    // Return gradients (only grad_value supported for now) should have 6 entries
    if (grads.size() < 6) {
      grads.resize(6);
    }

    return grads;   // matches the order of the forward arguments
};

// [[torch::export]]
torch::Tensor multiscale_deformable_attn(
    const torch::Tensor &value,
    const torch::Tensor &spatial_shapes,
    const torch::Tensor &level_start_index,
    const torch::Tensor &sampling_loc,
    const torch::Tensor &attn_weight,
    const int im2col_step
) {
  return MSDeformAttnFunction::apply(
    value, spatial_shapes, level_start_index,
    sampling_loc, attn_weight, im2col_step
  );
}
