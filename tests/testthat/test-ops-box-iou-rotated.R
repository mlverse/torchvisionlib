dtypes <- list(torch::torch_float32(), torch::torch_float64())

# cxcywhr boxes used by the core test, together with their expected pairwise
# IoU. Squares 1-4 are the same box rotated in 90-degree steps (IoU 1); box 5
# is far away (IoU 0); the 45/135-degree rectangles 6-8 overlap in a cross
# (IoU 1/3) or coincide (IoU 1). Ported from torchvision's TestRotatedBoxIou.
rotated_boxes <- torch::torch_tensor(matrix(c(
  0,   0,   10, 10, 45,
  0,   0,   10, 10, 135,
  0,   0,   10, 10, -45,
  0,   0,   10, 10, -135,
  100, 100, 10, 10, 30,
  50,  50,  20, 10, 45,
  50,  50,  20, 10, 135,
  50,  50,  20, 10, -135
), ncol = 5, byrow = TRUE))

rotated_boxes_expected <- torch::torch_tensor(matrix(c(
  1, 1, 1, 1, 0, 0,   0,   0,
  1, 1, 1, 1, 0, 0,   0,   0,
  1, 1, 1, 1, 0, 0,   0,   0,
  1, 1, 1, 1, 0, 0,   0,   0,
  0, 0, 0, 0, 1, 0,   0,   0,
  0, 0, 0, 0, 0, 1,   1/3, 1,
  0, 0, 0, 0, 0, 1/3, 1,   1/3,
  0, 0, 0, 0, 0, 1,   1/3, 1
), ncol = 8, byrow = TRUE), dtype = torch::torch_float32())

# cxcywhr -> xywhr / xyxyxyxy, used to check the R-side format conversions
# against the cxcywhr path. Formulas mirror torchvision's box_convert.
cxcywhr_to_xywhr <- function(boxes) {
  b <- torch::torch_unbind(boxes, dim = -1)
  cx <- b[[1]]
  cy <- b[[2]]
  w <- b[[3]]
  h <- b[[4]]
  r <- b[[5]]
  rad <- r * pi / 180
  cos <- torch::torch_cos(rad)
  sin <- torch::torch_sin(rad)
  x1 <- cx - w / 2 * cos - h / 2 * sin
  y1 <- cy - h / 2 * cos + w / 2 * sin
  torch::torch_stack(list(x1, y1, w, h, r), dim = -1)
}

xywhr_to_xyxyxyxy <- function(boxes) {
  b <- torch::torch_unbind(boxes, dim = -1)
  x1 <- b[[1]]
  y1 <- b[[2]]
  w <- b[[3]]
  h <- b[[4]]
  r <- b[[5]]
  rad <- r * pi / 180
  cos <- torch::torch_cos(rad)
  sin <- torch::torch_sin(rad)
  x2 <- x1 + w * cos
  y2 <- y1 - w * sin
  x3 <- x2 + h * sin
  y3 <- y2 + h * cos
  x4 <- x1 + h * sin
  y4 <- y1 + h * cos
  torch::torch_stack(list(x1, y1, x2, y2, x3, y3, x4, y4), dim = -1)
}

test_that("box_iou_rotated matches expected IoU for cxcywhr boxes", {
  for (dtype in dtypes) {
    boxes <- rotated_boxes$to(dtype = dtype)
    out <- ops_box_iou_rotated(boxes, boxes)
    # The op returns float32 regardless of the input dtype.
    expect_true(out$dtype == torch::torch_float32())
    expect_equal_to_tensor(out, rotated_boxes_expected, atol = 1e-4, rtol = 1e-4)
  }
})

test_that("box_iou_rotated is invariant to angle sign and 180-degree rotation", {
  for (dtype in dtypes) {
    base <- torch::torch_tensor(matrix(c(0, 0, 10, 10, 0), nrow = 1), dtype = dtype)
    for (angle in c(45, 53, 195)) {
      pos <- torch::torch_tensor(matrix(c(0, 0, 10, 10, angle), nrow = 1), dtype = dtype)
      neg <- torch::torch_tensor(matrix(c(0, 0, 10, 10, -angle), nrow = 1), dtype = dtype)
      expect_equal_to_tensor(
        ops_box_iou_rotated(base, pos), ops_box_iou_rotated(base, neg)
      )
    }

    rect <- torch::torch_tensor(matrix(c(50, 50, 20, 10, 30), nrow = 1), dtype = dtype)
    rect180 <- torch::torch_tensor(matrix(c(50, 50, 20, 10, 30 - 180), nrow = 1), dtype = dtype)
    expect_equal(as.numeric(ops_box_iou_rotated(rect, rect180)), 1, tolerance = 1e-5)
  }
})

# Deliberately float64: recovering the angle from corners via atan2 loses enough
# float32 precision that upstream needs atol = 0.5 for xyxyxyxy on macOS. Using
# float64 keeps this a tight check of the conversions rather than of float noise.
test_that("box_iou_rotated gives the same result across formats", {
  boxes <- rotated_boxes$to(dtype = torch::torch_float64())
  ref <- ops_box_iou_rotated(boxes, boxes, fmt = "cxcywhr")

  xywhr <- cxcywhr_to_xywhr(boxes)
  expect_equal_to_tensor(
    ops_box_iou_rotated(xywhr, xywhr, fmt = "xywhr"), ref, atol = 1e-4
  )

  xyxyxyxy <- xywhr_to_xyxyxyxy(xywhr)
  expect_equal_to_tensor(
    ops_box_iou_rotated(xyxyxyxy, xyxyxyxy, fmt = "xyxyxyxy"), ref, atol = 1e-4
  )
})

test_that("box_iou_rotated handles containment and zero-area boxes", {
  for (dtype in dtypes) {
    outer <- torch::torch_tensor(matrix(c(0, 0, 20, 20, 45), nrow = 1), dtype = dtype)
    inner <- torch::torch_tensor(matrix(c(0, 0, 10, 10, 45), nrow = 1), dtype = dtype)
    expect_equal(as.numeric(ops_box_iou_rotated(outer, inner)), 0.25, tolerance = 1e-5)

    degenerate <- torch::torch_tensor(matrix(c(0, 0, 0, 10, 45), nrow = 1), dtype = dtype)
    other <- torch::torch_tensor(matrix(c(0, 0, 10, 10, 30), nrow = 1), dtype = dtype)
    expect_equal(as.numeric(ops_box_iou_rotated(degenerate, other)), 0)
  }
})

test_that("box_iou_rotated handles empty inputs and output shape", {
  for (dtype in dtypes) {
    boxes <- torch::torch_zeros(c(10, 5), dtype = dtype)
    empty <- torch::torch_zeros(c(0, 5), dtype = dtype)
    expect_equal(ops_box_iou_rotated(empty, boxes)$shape, c(0, 10))
    expect_equal(ops_box_iou_rotated(boxes, empty)$shape, c(10, 0))

    b1 <- torch::torch_rand(5, 5)$to(dtype = dtype)
    b2 <- torch::torch_rand(7, 5)$to(dtype = dtype)
    expect_equal(ops_box_iou_rotated(b1, b2)$shape, c(5, 7))
  }
})

test_that("box_iou_rotated errors on an unknown format", {
  boxes <- torch::torch_tensor(matrix(c(0, 0, 10, 10, 0), nrow = 1))
  expect_error(
    ops_box_iou_rotated(boxes, boxes, fmt = "nope"),
    regexp = "Unsupported format"
  )
})

test_that("box_iou_rotated is numerically stable (Detectron2 regressions)", {
  for (dtype in dtypes) {
    # Precision at large coordinates: IoU is the height ratio.
    b1 <- torch::torch_tensor(matrix(c(565, 565, 10, 10, 0), nrow = 1), dtype = dtype)
    b2 <- torch::torch_tensor(matrix(c(565, 565, 10, 8.3, 0), nrow = 1), dtype = dtype)
    expect_equal(as.numeric(ops_box_iou_rotated(b1, b2)), 8.3 / 10, tolerance = 1e-4)

    # Nearly identical large boxes should have IoU close to 1.
    b3 <- torch::torch_tensor(matrix(c(2563.74462890625, 1436.7901611328125,
                                       2174.703369140625, 214.095001220703125,
                                       115.11834716796875), nrow = 1), dtype = dtype)
    b4 <- torch::torch_tensor(matrix(c(2563.74462890625, 1436.790283203125,
                                       2174.702880859375, 214.0949554443359375,
                                       115.11835479736328125), nrow = 1), dtype = dtype)
    expect_equal(as.numeric(ops_box_iou_rotated(b3, b4)), 1, tolerance = 1e-3)

    # Extreme coordinates must not push the IoU outside [0, 1].
    b5 <- torch::torch_tensor(matrix(c(160, 153, 230, 23, -37), nrow = 1), dtype = dtype)
    b6 <- torch::torch_tensor(matrix(c(-1.117407639806935e17, 1.3858420478349148e18,
                                       1000, 1000, 1612), nrow = 1), dtype = dtype)
    iou <- as.numeric(ops_box_iou_rotated(b5, b6))
    expect_gte(iou, 0)
    expect_lte(iou, 1)
  }
})
