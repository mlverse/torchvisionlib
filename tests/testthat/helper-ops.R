library(torch)

create_tensors_with_iou <- function(n, iou_thresh) {
  # force last box to have a pre-defined iou with the first box
  # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
  # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
  # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
  # Adjust the threshold upward a bit with the intent of creating
  # at least one box that exceeds (barely) the threshold and so
  # should be suppressed.
  boxes <- torch::torch_rand(n, 4) * 100
  boxes[, 3:N] <- boxes[, 3:N] + boxes[, 1:2]
  b <- as.numeric(boxes[-1]) #x0, y0, x1, y1
  iou_thresh <- iou_thresh + 1e-5
  boxes[-1, 3] <- boxes[-1, 3] + (b[3] - b[1]) * (1 - iou_thresh) / iou_thresh
  boxes
}

expect_tensor <- function(x) {
  expect_true(inherits(x, "torch_tensor"))
}

expect_equal_to_tensor <- function(x, y, ...) {
  expect_tensor(x)
  expect_tensor(y)
  expect_true(torch::torch_allclose(x, y, ...))
}
