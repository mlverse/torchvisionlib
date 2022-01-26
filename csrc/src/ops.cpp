#include <lantern/types.h>
#include <torchvision/ops/ops.h>
#include <torchvisionlib/torchvisionlib.h>

// [[torch::export]]
torch::Tensor vision_ops_nms(torch::Tensor dets, torch::Tensor scores, double iou_threshold) {
  return vision::ops::nms(dets, scores, iou_threshold);
}

// [[torch::export]]
torch::Tensor vision_ops_deform_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor offset,
    torch::Tensor mask,
    torch::Tensor bias,
    std::int64_t stride_h,
    std::int64_t stride_w,
    std::int64_t pad_h,
    std::int64_t pad_w,
    std::int64_t dilation_h,
    std::int64_t dilation_w,
    std::int64_t groups,
    std::int64_t offset_groups,
    bool use_mask) {
  return vision::ops::deform_conv2d(
    input,
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    offset_groups,
    use_mask
  );
}
