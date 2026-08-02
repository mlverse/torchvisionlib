#include <lantern/types.h>
#include <string>
#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include <torchvision/ops/ops.h>
#include <torchvision/io/image/cpu/decode_jpeg.h>
#include <torchvisionlib/torchvisionlib.h>
#include <torchvisionlib/torchvisionlib_types.h>
#include "ops/ms_deform_attn/ms_deform_attn.h"
#include "ops/roi_align_rotated/roi_align_rotated.h"

// [[torch::export]]
torch::Tensor vision_ops_nms(torch::Tensor dets, torch::Tensor scores, double iou_threshold) {
  return vision::ops::nms(dets, scores, iou_threshold);
}

// [[torch::export]]
torch::Tensor vision_ops_ms_deform_attn(
    torch::Tensor value,
    torch::Tensor spatial_shapes,
    torch::Tensor level_start_index,
    torch::Tensor sampling_loc,
    torch::Tensor attn_weight,
    std::int64_t im2col_step) {
  return vision::ops::ms_deform_attn(
    value,
    spatial_shapes,
    level_start_index,
    sampling_loc,
    attn_weight,
    im2col_step
  );
}

// [[torch::export]]
torch::Tensor vision_ops_roi_align_rotated(
    torch::Tensor input,
    torch::Tensor rois,
    std::int64_t pooled_height,
    std::int64_t pooled_width,
    double spatial_scale,
    std::int64_t sampling_ratio,
    bool aligned,
    bool clockwise) {
  return vision::ops::roi_align_rotated(
    input,
    rois,
    pooled_height,
    pooled_width,
    spatial_scale,
    sampling_ratio,
    aligned,
    clockwise
  );
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

// [[torch::export]]
tensor_pair vision_ops_ps_roi_pool(
    torch::Tensor input,
    torch::Tensor rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  return vision::ops::ps_roi_pool(
    input,
    rois,
    spatial_scale,
    pooled_height,
    pooled_width
  );
}

// [[torch::export]]
torch::Tensor vision_ops_roi_align(
    torch::Tensor input,
    torch::Tensor rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  return vision::ops::roi_align(
    input,
    rois,
    spatial_scale,
    pooled_height,
    pooled_width,
    sampling_ratio,
    aligned
  );
}

// [[torch::export]]
tensor_pair vision_ops_roi_pool(
    torch::Tensor input,
    torch::Tensor rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  return vision::ops::roi_pool(
    input,
    rois,
    spatial_scale,
    pooled_height,
    pooled_width
  );
}

// [[torch::export]]
torch::Tensor vision_read_jpeg(std::string fpath) {
  std::ifstream file(fpath, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size))
  {
      throw std::runtime_error("Error reading file.");
  }

  auto ten = torch::from_blob(buffer.data(), {size}, torch::kByte);
  return vision::image::decode_jpeg(ten);
}

// [[torch::export]]
torch::Tensor vision_read_jpeg_float(std::string fpath) {
  auto ten = vision_read_jpeg(fpath);
  return ten.to(torch::kFloat32).div_(255);
}
