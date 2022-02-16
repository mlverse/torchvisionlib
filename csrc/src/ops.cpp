#include <lantern/types.h>
#include <torchvision/ops/ops.h>
#include <torchvisionlib/torchvisionlib.h>
#include <torchvisionlib/torchvisionlib_types.h>

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

// [[torch::export(register_types=c("tensor_pair", "TensorPair", "void*", "torchvisionlib::tensor_pair"))]]
tensor_pair vision_ops_ps_roi_align(
    torch::Tensor input,
    torch::Tensor rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio) {
  return vision::ops::ps_roi_align(
    input,
    rois,
    spatial_scale,
    pooled_height,
    pooled_width,
    sampling_ratio
  );
};
