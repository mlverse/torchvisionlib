
#ifndef MS_DEFORM_ATTN_AUTOGRAD_H
#define MS_DEFORM_ATTN_AUTOGRAD_H

#pragma once

#include <torch/torch.h>

// Declaration of the autograd function
struct MSDeformAttnFunction : torch::autograd::Function<MSDeformAttnFunction> {
  public:
    static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& value,
      const torch::Tensor& spatial_shapes,
      const torch::Tensor& level_start_index,
      const torch::Tensor& sampling_loc,
      const torch::Tensor& attn_weight,
      const int im2col_step
  );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor &grad_output);
};

// Public interface function
torch::Tensor multiscale_deformable_attn(
    const torch::Tensor& value,
    const torch::Tensor& spatial_shapes,
    const torch::Tensor& level_start_index,
    const torch::Tensor& sampling_loc,
    const torch::Tensor& attn_weight,
    const int im2col_step = 1
);

#endif // MS_DEFORM_ATTN_AUTOGRAD_H
