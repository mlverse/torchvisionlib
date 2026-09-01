library(torch)

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
    matrix(c(1, 3, 3, 5, 5, 0.4, 2, 3, 4, 4, 4, -0.2), ncol = 6, byrow = TRUE),
    dtype = torch_float32()
  )
  output_size <- c(3, 3)

  out <- ops_roi_align_rotated(input, rois, output_size, spatial_scale = 1,
                               sampling_ratio = 2)
  out$sum()$backward()
  expect_true(!is.null(input$grad))

  eps <- 1e-3
  num_grad <- torch_empty_like(input)
  num_grad_flat <- num_grad$flatten()
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
    num_grad_flat[i] <- as.numeric(f_p - f_m) / (2 * eps)
  }

  expect_true(torch_allclose(input$grad, num_grad, atol = 1e-3, rtol = 1e-3))
})

test_that("nn_roi_align_rotated module works", {
  input <- torch_randn(1, 2, 10, 10)
  rois <- torch_tensor(matrix(c(1, 5, 5, 4, 4, 0.3), ncol = 6),
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
  neg_rois <- torch_tensor(matrix(c(1, 3, 3, -2, 5, 0.4), ncol = 6),
                           dtype = torch_float32())
  expect_error(
    ops_roi_align_rotated(input, neg_rois, c(3, 3), 1),
    regexp = "do not have non-negative size"
  )

  # out-of-range batch index
  oob_rois <- torch_tensor(matrix(c(0, 3, 3, 5, 5, 0.4), ncol = 6),
                           dtype = torch_float32())
  expect_error(
    ops_roi_align_rotated(input, oob_rois, c(3, 3), 1),
    regexp = "rois index should be in \\[1, batch_size\\]"
  )

  # non-positive output size
  expect_error(
    ops_roi_align_rotated(input, rois, c(0, 3), 1),
    regexp = "pooled_height and pooled_width should be positive"
  )
})
