# Standalone test for ops_roi_align_rotated (mmcv roi_align_rotated, CPU).
#
# Run after installing torchvisionlib on Windows:
#   Rscript tests/run-roi-align-rotated.R
#
# Prints PASS/FAIL for each check and exits with status 0/1.

suppressMessages({ library(torch); library(torchvisionlib) })

pass <- 0L
fail <- 0L

check <- function(label, cond) {
  if (isTRUE(cond)) {
    pass <<- pass + 1L
    cat(sprintf("PASS  %s\n", label))
  } else {
    fail <<- fail + 1L
    cat(sprintf("FAIL  %s\n", label))
  }
}

# ---------------------------------------------------------------------------
# Pure-R reference of the mmcv CPU kernel
# ---------------------------------------------------------------------------
bilinear_sample <- function(feat, ys, xs) {
  H <- dim(feat)[1]; W <- dim(feat)[2]
  n <- length(ys)
  out <- numeric(n)
  for (i in seq_len(n)) {
    y <- ys[i]; x <- xs[i]
    if (y < -1 || y > H || x < -1 || x > W) { out[i] <- 0; next }
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
  N <- dim(input)[1]; C <- dim(input)[2]; H <- dim(input)[3]; W <- dim(input)[4]
  out_h <- output_size[1]; out_w <- output_size[2]
  K <- dim(rois)[1]

  input_r <- as.array(input)
  rois_m <- matrix(as.numeric(rois), ncol = 6)
  out <- array(0, dim = c(K, C, out_h, out_w))

  for (n in seq_len(K)) {
    roi <- rois_m[n, ]
    roi_batch_ind <- as.integer(roi[1]) + 1L
    offset <- if (aligned) 0.5 else 0.0
    roi_center_w <- roi[2] * spatial_scale - offset
    roi_center_h <- roi[3] * spatial_scale - offset
    roi_width <- roi[4] * spatial_scale
    roi_height <- roi[5] * spatial_scale
    theta <- roi[6]
    if (clockwise) theta <- -theta
    cos_theta <- cos(theta)
    sin_theta <- sin(theta)
    if (!aligned) { roi_width <- max(roi_width, 1); roi_height <- max(roi_height, 1) }
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

# ---------------------------------------------------------------------------
# 1. Forward matches the reference
# ---------------------------------------------------------------------------
torch::torch_manual_seed(42)
input <- torch_randn(2, 3, 8, 8)
rois <- torch_tensor(
  matrix(c(
    0, 3.5, 3.5, 5, 5, 0.5,      # batch 0, ~centered box
    1, 4.2, 4.8, 6, 4, -0.3,     # batch 1, wider than tall
    0, 2.0, 2.0, 3, 3, pi / 4    # batch 0, small rotated box
  ), ncol = 6, byrow = TRUE),
  dtype = torch_float32()
)

out <- ops_roi_align_rotated(input, rois, c(3, 3), spatial_scale = 1,
                             sampling_ratio = 2, aligned = TRUE)
ref <- roi_align_rotated_reference(input, rois, c(3, 3), 1, 2, TRUE)
check("forward aligned=TRUE  shape",
      identical(dim(out), c(3L, 3L, 3L, 3L)))
check("forward aligned=TRUE  vs reference",
      torch_allclose(out, torch_tensor(ref), atol = 1e-5, rtol = 1e-5))

out2 <- ops_roi_align_rotated(input, rois, c(2, 4), spatial_scale = 0.5,
                              sampling_ratio = 0, aligned = FALSE, clockwise = TRUE)
ref2 <- roi_align_rotated_reference(input, rois, c(2, 4), 0.5, 0, FALSE, TRUE)
check("forward aligned=FALSE/clockwise shape",
      identical(dim(out2), c(3L, 3L, 2L, 4L)))
check("forward aligned=FALSE/clockwise vs reference",
      torch_allclose(out2, torch_tensor(ref2), atol = 1e-5, rtol = 1e-5))

# ---------------------------------------------------------------------------
# 2. Gradients (autograd) match finite differences
# ---------------------------------------------------------------------------
torch::torch_manual_seed(1)
input_g <- torch_randn(2, 2, 6, 6, requires_grad = TRUE)
rois_g <- torch_tensor(
  matrix(c(0, 3, 3, 5, 5, 0.4, 1, 3, 4, 4, 4, -0.2), ncol = 6, byrow = TRUE),
  dtype = torch_float32()
)
output_size <- c(3, 3)

ops_roi_align_rotated(input_g, rois_g, output_size, spatial_scale = 1,
                      sampling_ratio = 2)$sum()$backward()
check("backward produced grad", !is.null(input_g$grad))

eps <- 1e-3
num_grad <- torch_empty_like(input_g)
input_flat <- input_g$detach()$flatten()
for (i in seq_len(prod(dim(input_g)))) {
  x_p <- input_flat$clone(); x_m <- input_flat$clone()
  x_p[i] <- as.numeric(x_p[i]) + eps
  x_m[i] <- as.numeric(x_m[i]) - eps
  f_p <- ops_roi_align_rotated(x_p$view(dim(input_g)), rois_g, output_size, 1,
                               sampling_ratio = 2)$sum()
  f_m <- ops_roi_align_rotated(x_m$view(dim(input_g)), rois_g, output_size, 1,
                               sampling_ratio = 2)$sum()
  num_grad$flatten()[i] <- as.numeric(f_p - f_m) / (2 * eps)
}
check("backward matches finite differences",
      torch_allclose(input_g$grad, num_grad, atol = 1e-3, rtol = 1e-3))

# ---------------------------------------------------------------------------
# 3. nn module
# ---------------------------------------------------------------------------
mod <- nn_roi_align_rotated(output_size = c(4, 4), spatial_scale = 1, sampling_ratio = 1)
out_mod <- mod(torch_randn(1, 2, 10, 10),
               torch_tensor(matrix(c(0, 5, 5, 4, 4, 0.3), ncol = 6), dtype = torch_float32()))
check("nn_roi_align_rotated output shape", identical(dim(out_mod), c(1L, 2L, 4L, 4L)))

# ---------------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------------
ok_err <- function(expr, pattern) {
  res <- tryCatch({ force(expr); "no error" }, error = function(e) conditionMessage(e))
  grepl(pattern, res, fixed = TRUE)
}
check("error: rois must have 6 columns",
      ok_err(
        ops_roi_align_rotated(input, torch_tensor(matrix(c(0, 3, 3, 5, 5), ncol = 5), dtype = torch_float32()), c(3, 3), 1),
        "rois should have 6 columns"))
check("error: negative roi size when aligned",
      ok_err(
        ops_roi_align_rotated(input, torch_tensor(matrix(c(0, 3, 3, -2, 5, 0.4), ncol = 6), dtype = torch_float32()), c(3, 3), 1),
        "do not have non-negative size"))
check("error: out-of-range batch index",
      ok_err(
        ops_roi_align_rotated(input, torch_tensor(matrix(c(5, 3, 3, 5, 5, 0.4), ncol = 6), dtype = torch_float32()), c(3, 3), 1),
        "rois index should be in [0, batch_size)"))
check("error: non-positive output size",
      ok_err(ops_roi_align_rotated(input, rois, c(0, 3), 1),
             "pooled_height and pooled_width should be positive"))

# ---------------------------------------------------------------------------
cat(sprintf("\n%d passed, %d failed\n", pass, fail))
if (fail > 0) quit(status = 1)
