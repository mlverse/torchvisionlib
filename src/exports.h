// Generated by using torchexport::export() -> do not edit by hand
#include <Rcpp.h>
#include <torch.h>
#include "torchvisionlib_types.h"

torch::Tensor rcpp_vision_ops_nms (torch::Tensor dets, torch::Tensor scores, double iou_threshold);
torch::Tensor rcpp_vision_ops_deform_conv2d (torch::Tensor input, torch::Tensor weight, torch::Tensor offset, torch::Tensor mask, torch::Tensor bias, std::int64_t stride_h, std::int64_t stride_w, std::int64_t pad_h, std::int64_t pad_w, std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t groups, std::int64_t offset_groups, bool use_mask);
torchvisionlib::tensor_pair rcpp_vision_ops_ps_roi_align (torch::Tensor input, torch::Tensor rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio);
void rcpp_delete_tensor_pair (void* x);
torch::Tensor rcpp_tensor_pair_get_first (torchvisionlib::tensor_pair x);
torch::Tensor rcpp_tensor_pair_get_second (torchvisionlib::tensor_pair x);