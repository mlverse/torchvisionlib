
#include "ms_deform_attn.h"
#include <torchvisionlib/exports.h>  // from torchexport

// Autograd function
struct MSDeformAttnFunction : torch::autograd::Function<MSDeformAttnFunction> {
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& value,
      const torch::Tensor& spatial_shapes,
      const torch::Tensor& level_start_index,
      const torch::Tensor& sampling_loc,
      const torch::Tensor& attn_weight,
      int64_t im2col_step
  ) {
    ctx->saved_data["spatial_shapes"] = spatial_shapes;
    ctx->saved_data["level_start_index"] = level_start_index;
    ctx->saved_data["im2col_step"] = im2col_step;
    ctx->save_for_backward({value, sampling_loc, attn_weight});

    auto result = torch::jit::getPackedTensorImpl(
      torch::jit::invoke(
        "tvision::ms_deform_attn_forward",
        value, spatial_shapes, level_start_index,
        sampling_loc, attn_weight, im2col_step
      )
    );

    return torch::make_tensor(result);
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs
  ) {
    auto grad_output = grad_outputs[0].contiguous();
    auto saved = ctx->get_saved_variables();
    auto value = saved[0];
    auto sampling_loc = saved[1];
    auto attn_weight = saved[2];

    auto spatial_shapes = ctx->saved_data["spatial_shapes"].toTensor();
    auto level_start_index = ctx->saved_data["level_start_index"].toTensor();
    auto im2col_step = ctx->saved_data["im2col_step"].toInt();

    auto grad_input = torch::jit::invoke(
      "tvision::ms_deform_attn_backward",
      grad_output, value, spatial_shapes,
      level_start_index, sampling_loc, attn_weight, im2col_step
    ).toTensor();

    // Return gradients (only grad_value supported for now)
    return {
      grad_input,
      torch::Tensor(),  // grad_spatial_shapes
      torch::Tensor(),  // grad_level_start_index
      torch::Tensor(),  // grad_sampling_loc
      torch::Tensor(),  // grad_attn_weight
      torch::Tensor()   // grad_im2col_step
    };
  }
};

// [[torch::export]]
torch::Tensor multiscale_deformable_attn(
    const torch::Tensor& value,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index,
    const torch::Tensor& sampling_loc,
    const torch::Tensor& attn_weight,
    int im2col_step = 64
) {
  return MSDeformAttnFunction::apply(
    value, spatial_shapes, level_start_index,
    sampling_loc, attn_weight, im2col_step
  );
}
