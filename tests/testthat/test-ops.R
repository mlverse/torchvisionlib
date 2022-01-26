test_that("nms", {
  torch::torch_manual_seed(1)
  boxes <- create_tensors_with_iou(10, 1)
  scores <- torch::torch_rand(10)
  result <- ops_nms(boxes, scores, 1)
  reference <- torch::torch_tensor(as.integer(c(0, 2, 5, 7, 3, 9, 8, 1, 4, 6) + 1))
  expect_equal_to_tensor(result, reference)

  expect_error(
    ops_nms(torch::torch_rand(4), torch::torch_rand(3), 0.5),
    regexp = "boxes should be a 2d tensor, got 1D"
  )
  expect_error(
    ops_nms(torch::torch_rand(3, 5), torch::torch_rand(3), 0.5),
    regexp = "boxes should have 4 elements in dimension 1, got 5"
  )
  expect_error(
    ops_nms(torch::torch_rand(3, 4), torch::torch_rand(3,2), 0.5),
    regexp = "scores should be a 1d tensor, got 2D"
  )
  expect_error(
    ops_nms(torch::torch_rand(3, 4), torch::torch_rand(4), 0.5),
    regexp = "boxes and scores should have same number of elements in dimension 0, got 3 and 4"
  )

})

test_that("deform_conv", {

  input <- torch_rand(4, 3, 10, 10)
  kh <- kw <- 3
  weight <- torch_rand(5, 3, kh, kw)
  offset <- torch_rand(4, 2 * kh * kw, 8, 8)
  mask <- torch_rand(4, kh * kw, 8, 8)
  out <- ops_deform_conv2d(input, offset, weight, mask = mask)
  expect_equal(out$shape, c(4,5,8,8))

  expect_error(
    ops_deform_conv2d(torch_rand(10), offset, weight, mask = mask),
    regexp = "Expected input_c.ndimension()"
  )
  expect_error(
    ops_deform_conv2d(input, torch_rand(5, 4, kh, kw), weight, mask = mask),
    regexp = "the shape of the offset tensor at"
  )

})
