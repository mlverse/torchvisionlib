// Generated by using torchexport::export() -> do not edit by hand
#include <Rcpp.h>
#include <torch.h>
#define TORCHVISIONLIB_HEADERS_ONLY
#include <torchvisionlib/torchvisionlib.h>

// [[Rcpp::export]]
torch::Tensor rcpp_vision_ops_nms (torch::Tensor dets, torch::Tensor scores, double iou_threshold) {
  return  vision_ops_nms(dets.get(), scores.get(), iou_threshold);
}