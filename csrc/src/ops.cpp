#include <lantern/types.h>
#include <torchvision/ops/ops.h>
#include "vision/vision.h"

R_VISION_API void* c_vision_ops_nms (void* dets, void* scores, double iou_threshold) {
    return make_raw::Tensor(vision::ops::nms(
        from_raw::Tensor(dets),
        from_raw::Tensor(scores),
        iou_threshold
    ));
}

