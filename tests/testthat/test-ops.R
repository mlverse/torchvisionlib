test_that("nms", {
  torch::torch_manual_seed(1)
  boxes <- create_tensors_with_iou(10, 1)
  scores <- torch::torch_rand(10)
  result <- ops_nms(boxes, scores, 1)
  reference <- torch::torch_tensor(as.integer(c(0, 2, 5, 7, 3, 9, 8, 1, 4, 6) + 1))
  expect_equal_to_tensor(result, reference)
})
