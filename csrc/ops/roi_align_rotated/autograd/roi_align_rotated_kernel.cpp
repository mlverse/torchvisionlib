#include "../roi_align_rotated.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace vision {
namespace ops {

namespace {

class ROIAlignRotatedFunction
    : public torch::autograd::Function<ROIAlignRotatedFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& input,
      const torch::autograd::Variable& rois,
      int64_t pooled_height,
      int64_t pooled_width,
      double spatial_scale,
      int64_t sampling_ratio,
      bool aligned,
      bool clockwise) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto output = roi_align_rotated(
        input,
        rois,
        pooled_height,
        pooled_width,
        spatial_scale,
        sampling_ratio,
        aligned,
        clockwise);

    ctx->save_for_backward({rois});
    ctx->saved_data["pooled_height"] = pooled_height;
    ctx->saved_data["pooled_width"] = pooled_width;
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["sampling_ratio"] = sampling_ratio;
    ctx->saved_data["aligned"] = aligned;
    ctx->saved_data["clockwise"] = clockwise;
    ctx->saved_data["batch_size"] = input.size(0);
    ctx->saved_data["channels"] = input.size(1);
    ctx->saved_data["height"] = input.size(2);
    ctx->saved_data["width"] = input.size(3);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    auto saved = ctx->get_saved_variables();
    auto rois = saved[0];

    auto grad_input = detail::_roi_align_rotated_backward(
        grad_output[0],
        rois,
        ctx->saved_data["pooled_height"].toInt(),
        ctx->saved_data["pooled_width"].toInt(),
        ctx->saved_data["spatial_scale"].toDouble(),
        ctx->saved_data["sampling_ratio"].toInt(),
        ctx->saved_data["aligned"].toBool(),
        ctx->saved_data["clockwise"].toBool(),
        ctx->saved_data["batch_size"].toInt(),
        ctx->saved_data["channels"].toInt(),
        ctx->saved_data["height"].toInt(),
        ctx->saved_data["width"].toInt());

    return {
        grad_input,
        torch::autograd::Variable(), // rois
        torch::autograd::Variable(), // pooled_height
        torch::autograd::Variable(), // pooled_width
        torch::autograd::Variable(), // spatial_scale
        torch::autograd::Variable(), // sampling_ratio
        torch::autograd::Variable(), // aligned
        torch::autograd::Variable(), // clockwise
    };
  }
};

at::Tensor roi_align_rotated_autograd(
    const at::Tensor& input,
    const at::Tensor& rois,
    int64_t pooled_height,
    int64_t pooled_width,
    double spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    bool clockwise) {
  return ROIAlignRotatedFunction::apply(
      input,
      rois,
      pooled_height,
      pooled_width,
      spatial_scale,
      sampling_ratio,
      aligned,
      clockwise)[0];
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align_rotated"),
      TORCH_FN(roi_align_rotated_autograd));
}

} // namespace ops
} // namespace vision
