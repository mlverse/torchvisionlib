#include <lantern/types.h>
#include <torchvision/ops/ops.h>
#include <torchvisionlib/torchvisionlib.h>

// [[torch::export]]
torch::Tensor vision_ops_nms(torch::Tensor dets, torch::Tensor scores, double iou_threshold) {
  return vision::ops::nms(dets, scores, iou_threshold);
}

