test_that("C++ functions are properly exported", {
  # Test that the functions exist
  expect_true(exists("ms_deform_attn_forward_wrapper"))
  expect_true(exists("ms_deform_attn_backward_wrapper"))

  # Test that they are callable
  expect_type(ms_deform_attn_forward_wrapper, "closure")
  expect_type(ms_deform_attn_backward_wrapper, "closure")
})


test_that("ms_deform_attn_forward works with CPU tensors", {
  skip_on_cran() # Skip on CRAN if needed

  # Create simple test tensors
  value <- torch_randn(1, 10, 1, 64)  # batch_size, num_keys, num_levels, embed_dim
  spatial_shapes <- torch_tensor(matrix(c(2, 5), nrow = 1, ncol = 2))  # num_levels x 2
  level_start_index <- torch_tensor(c(0L))  # num_levels
  sampling_loc <- torch_randn(1, 10, 1, 1, 2)  # batch_size, num_queries, num_levels, num_points, 2
  attn_weight <- torch_randn(1, 10, 1, 1)  # batch_size, num_queries, num_levels, num_points

  # Test function call
  expect_silent({
    result <- ms_deform_attn_forward_wrapper(
      value, spatial_shapes, level_start_index,
      sampling_loc, attn_weight, 1L
    )
  })

  # Test result type
  expect_s3_class(result, "torch_tensor")

  # Test result dimensions (adjust based on your implementation)
  expect_equal(result$shape, c(1, 10, 1, 64))
})

test_that("ms_deform_attn_forward works with CUDA tensors", {
  skip_on_cran()
  skip_if_no_cuda()

  # Create CUDA tensors
  value <- torch_randn(1, 10, 1, 64)$cuda()
  spatial_shapes <- torch_tensor(matrix(c(2, 5), nrow = 1, ncol = 2))$cuda()
  level_start_index <- torch_tensor(c(0L))$cuda()
  sampling_loc <- torch_randn(1, 10, 1, 1, 2)$cuda()
  attn_weight <- torch_randn(1, 10, 1, 1)$cuda()

  expect_silent({
    result <- ms_deform_attn_forward_wrapper(
      value, spatial_shapes, level_start_index,
      sampling_loc, attn_weight, 1L
    )
  })

  expect_s3_class(result, "torch_tensor")
  expect_true(result$is_cuda())
})

test_that("ms_deform_attn_backward works correctly", {
  skip_on_cran()

  # Create test tensors
  value <- torch_randn(1, 10, 1, 64, requires_grad = TRUE)
  spatial_shapes <- torch_tensor(matrix(c(2, 5), nrow = 1, ncol = 2))
  level_start_index <- torch_tensor(c(0L))
  sampling_loc <- torch_randn(1, 10, 1, 1, 2)
  attn_weight <- torch_randn(1, 10, 1, 1, requires_grad = TRUE)
  grad_output <- torch_randn(1, 10, 1, 64)

  expect_silent({
    result <- ms_deform_attn_backward_wrapper(
      grad_output, value, spatial_shapes, level_start_index,
      sampling_loc, attn_weight, 1L
    )
  })

  # Test that result is a list/vector of tensors
  expect_type(result, "list")
  expect_gt(length(result), 0)

  # Test that each element is a tensor
  for (i in seq_along(result)) {
    expect_s3_class(result[[i]], "torch_tensor")
  }
})

test_that("ms_deform_attn_backward gradient computation works", {
  skip_on_cran()

  # Test gradient flow
  value <- torch_randn(1, 5, 1, 32, requires_grad = TRUE)
  spatial_shapes <- torch_tensor(matrix(c(1, 5), nrow = 1, ncol = 2))
  level_start_index <- torch_tensor(c(0L))
  sampling_loc <- torch_randn(1, 3, 1, 1, 2, requires_grad = TRUE)
  attn_weight <- torch_randn(1, 3, 1, 1, requires_grad = TRUE)

  # Forward pass
  output <- ms_deform_attn_forward_wrapper(
    value, spatial_shapes, level_start_index,
    sampling_loc, attn_weight, 1L
  )

  # Compute loss and backward
  loss <- output$sum()

  expect_silent({
    grads <- ms_deform_attn_backward_wrapper(
      torch_ones_like(output), value, spatial_shapes, level_start_index,
      sampling_loc, attn_weight, 1L
    )
  })

  # Check that gradients were computed
  expect_true(value$grad$is_not_null())
  expect_true(attn_weight$grad$is_not_null())
})
