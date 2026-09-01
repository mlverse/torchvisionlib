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

bilinear_sample <- function(feat, ys, xs) {
  H <- dim(feat)[1]; W <- dim(feat)[2]
  n <- length(ys)
  out <- numeric(n)
  for (i in seq_len(n)) {
    y <- ys[i]; x <- xs[i]
    if (y < -1 || y > H || x < -1 || x > W) {
      out[i] <- 0
      next
    }
    y <- max(y, 0); x <- max(x, 0)
    y_low <- floor(y); x_low <- floor(x)
    if (y_low >= H - 1) { y_high <- y_low; y_low <- H - 1; y <- H - 1 } else { y_high <- y_low + 1 }
    if (x_low >= W - 1) { x_high <- x_low; x_low <- W - 1; x <- W - 1 } else { x_high <- x_low + 1 }
    ly <- y - y_low; lx <- x - x_low
    hy <- 1 - ly; hx <- 1 - lx
    out[i] <- hy * hx * feat[y_low + 1, x_low + 1] +
      hy * lx * feat[y_low + 1, x_high + 1] +
      ly * hx * feat[y_high + 1, x_low + 1] +
      ly * lx * feat[y_high + 1, x_high + 1]
  }
  out
}

roi_align_rotated_reference <- function(input, rois, output_size, spatial_scale,
                                        sampling_ratio = 0, aligned = TRUE,
                                        clockwise = FALSE) {
  stopifnot(length(dim(input)) == 4)
  N <- dim(input)[1]; C <- dim(input)[2]; H <- dim(input)[3]; W <- dim(input)[4]
  out_h <- output_size[1]; out_w <- output_size[2]
  K <- dim(rois)[1]

  input_r <- as.array(input)
  rois_m <- matrix(as.numeric(rois), ncol = 6)
  out <- array(0, dim = c(K, C, out_h, out_w))

  for (n in seq_len(K)) {
    roi <- rois_m[n, ]
    roi_batch_ind <- as.integer(roi[1]) # already 1-based
    offset <- if (aligned) 0.5 else 0.0
    roi_center_w <- roi[2] * spatial_scale - offset
    roi_center_h <- roi[3] * spatial_scale - offset
    roi_width <- roi[4] * spatial_scale
    roi_height <- roi[5] * spatial_scale
    theta <- roi[6]
    if (clockwise) theta <- -theta
    cos_theta <- cos(theta)
    sin_theta <- sin(theta)
    if (!aligned) {
      roi_width <- max(roi_width, 1)
      roi_height <- max(roi_height, 1)
    }
    bin_size_h <- roi_height / out_h
    bin_size_w <- roi_width / out_w
    grid_h <- if (sampling_ratio > 0) sampling_ratio else ceiling(roi_height / out_h)
    grid_w <- if (sampling_ratio > 0) sampling_ratio else ceiling(roi_width / out_w)
    count <- max(grid_h * grid_w, 1)
    roi_start_h <- -roi_height / 2
    roi_start_w <- -roi_width / 2

    ys <- xs <- numeric(0)
    for (ph in seq_len(out_h)) {
      for (pw in seq_len(out_w)) {
        for (iy in seq_len(grid_h)) {
          yy <- roi_start_h + (ph - 1) * bin_size_h + (iy - 0.5) * bin_size_h / grid_h
          for (ix in seq_len(grid_w)) {
            xx <- roi_start_w + (pw - 1) * bin_size_w + (ix - 0.5) * bin_size_w / grid_w
            y <- yy * cos_theta - xx * sin_theta + roi_center_h
            x <- yy * sin_theta + xx * cos_theta + roi_center_w
            ys <- c(ys, y)
            xs <- c(xs, x)
          }
        }
      }
    }

    for (c in seq_len(C)) {
      vals <- bilinear_sample(input_r[roi_batch_ind, c, , ], ys, xs)
      means <- colMeans(matrix(vals, nrow = grid_h * grid_w))
      out[n, c, , ] <- matrix(means, nrow = out_h, byrow = TRUE)
    }
  }
  out
}

make_rois <- function() {
  torch::torch_tensor(
    matrix(c(
      1, 3.5, 3.5, 5, 5, 0.5,     # batch 1, ~centered box
      2, 4.2, 4.8, 6, 4, -0.3,    # batch 2, wider than tall
      1, 2.0, 2.0, 3, 3, pi / 4   # batch 1, small rotated box
    ), ncol = 6, byrow = TRUE),
    dtype = torch::torch_float32()
  )
}

make_input <- function() {
  torch::torch_manual_seed(42)
  torch::torch_randn(2, 3, 8, 8)
}
