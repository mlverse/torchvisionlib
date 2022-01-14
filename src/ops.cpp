#include <Rcpp.h>
#include <iostream>
#define VISION_HEADERS_ONLY
#include "vision/vision.h"
#include <torch.h>

// [[Rcpp::export]]
torch::index::Tensor vision_ops_nms (torch::Tensor dets, torch::Tensor scores,
                                     double iou_threshold) {
  return c_vision_ops_nms(dets.get(), scores.get(), iou_threshold);
}
