library(torch)

# Pure-R reference implementation of mmcv's `roi_align_rotated` CPU kernel
# (https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/csrc/pytorch/cpu/roi_align_rotated_cpu.cpp).
# Used to validate the C++ implementation independently.

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
    roi_batch_ind <- as.integer(roi[1]) + 1L # 0-based in C++, 1-based in R
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
  torch_tensor(
    matrix(c(
      0, 3.5, 3.5, 5, 5, 0.5,     # batch 0, ~centered box
      1, 4.2, 4.8, 6, 4, -0.3,    # batch 1, wider than tall
      0, 2.0, 2.0, 3, 3, pi / 4   # batch 0, small rotated box
    ), ncol = 6, byrow = TRUE),
    dtype = torch_float32()
  )
}

make_input <- function() {
  torch::torch_manual_seed(42)
  torch_randn(2, 3, 8, 8)
}

test_that("roi_align_rotated matches the mmcv reference (forward)", {
  input <- make_input()
  rois <- make_rois()

  out <- ops_roi_align_rotated(input, rois, c(3, 3), spatial_scale = 1,
                               sampling_ratio = 2, aligned = TRUE)
  ref <- roi_align_rotated_reference(input, rois, c(3, 3), 1, 2, TRUE)
  expect_equal(dim(out), c(3, 3, 3, 3))
  expect_true(torch_allclose(out, torch_tensor(ref), atol = 1e-5, rtol = 1e-5))
})

test_that("roi_align_rotated matches with aligned=FALSE and sampling_ratio=0", {
  input <- make_input()
  rois <- make_rois()

  out <- ops_roi_align_rotated(input, rois, c(2, 4), spatial_scale = 0.5,
                               sampling_ratio = 0, aligned = FALSE,
                               clockwise = TRUE)
  ref <- roi_align_rotated_reference(input, rois, c(2, 4), 0.5, 0, FALSE, TRUE)
  expect_equal(dim(out), c(3, 3, 2, 4))
  expect_true(torch_allclose(out, torch_tensor(ref), atol = 1e-5, rtol = 1e-5))
})

test_that("roi_align_rotated is differentiable and matches finite differences", {
  input <- torch::torch_randn(2, 2, 6, 6, requires_grad = TRUE)
  rois <- torch_tensor(
    matrix(c(0, 3, 3, 5, 5, 0.4, 1, 3, 4, 4, 4, -0.2), ncol = 6, byrow = TRUE),
    dtype = torch_float32()
  )
  output_size <- c(3, 3)

  out <- ops_roi_align_rotated(input, rois, output_size, spatial_scale = 1,
                               sampling_ratio = 2)
  out$sum()$backward()
  expect_true(!is.null(input$grad))

  eps <- 1e-3
  num_grad <- torch_empty_like(input)
  input_flat <- input$detach()$flatten()
  for (i in seq_len(numel <- prod(dim(input)))) {
    x_p <- input_flat$clone()
    x_m <- input_flat$clone()
    x_p[i] <- as.numeric(x_p[i]) + eps
    x_m[i] <- as.numeric(x_m[i]) - eps
    f_p <- ops_roi_align_rotated(x_p$view(dim(input)), rois, output_size, 1,
                                 sampling_ratio = 2)$sum()
    f_m <- ops_roi_align_rotated(x_m$view(dim(input)), rois, output_size, 1,
                                 sampling_ratio = 2)$sum()
    num_grad$flatten()[i] <- as.numeric(f_p - f_m) / (2 * eps)
  }

  expect_true(torch_allclose(input$grad, num_grad, atol = 1e-3, rtol = 1e-3))
})

test_that("nn_roi_align_rotated module works", {
  input <- torch_randn(1, 2, 10, 10)
  rois <- torch_tensor(matrix(c(0, 5, 5, 4, 4, 0.3), ncol = 6),
                       dtype = torch_float32())
  mod <- nn_roi_align_rotated(output_size = c(4, 4), spatial_scale = 1,
                              sampling_ratio = 1)
  out <- mod(input, rois)
  expect_equal(dim(out), c(1, 2, 4, 4))
})

test_that("roi_align_rotated validates its inputs", {
  input <- make_input()
  rois <- make_rois()

  # wrong number of rois columns
  bad_rois <- torch_tensor(matrix(c(0, 3, 3, 5, 5), ncol = 5),
                           dtype = torch_float32())
  expect_error(
    ops_roi_align_rotated(input, bad_rois, c(3, 3), 1),
    regexp = "rois should have 6 columns"
  )

  # negative box size with aligned = TRUE
  neg_rois <- torch_tensor(matrix(c(0, 3, 3, -2, 5, 0.4), ncol = 6),
                           dtype = torch_float32())
  expect_error(
    ops_roi_align_rotated(input, neg_rois, c(3, 3), 1),
    regexp = "do not have non-negative size"
  )

  # out-of-range batch index
  oob_rois <- torch_tensor(matrix(c(5, 3, 3, 5, 5, 0.4), ncol = 6),
                           dtype = torch_float32())
  expect_error(
    ops_roi_align_rotated(input, oob_rois, c(3, 3), 1),
    regexp = "rois index should be in \\[0, batch_size\\)"
  )

  # non-positive output size
  expect_error(
    ops_roi_align_rotated(input, rois, c(0, 3), 1),
    regexp = "pooled_height and pooled_width should be positive"
  )
})
