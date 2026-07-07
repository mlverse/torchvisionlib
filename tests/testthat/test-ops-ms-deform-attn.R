library(torch)

# Independent pure-R reference, equivalent to Deformable-DETR's
# `ms_deform_attn_core_pytorch`. Used to validate the CUDA kernel.
ms_deform_attn_reference <- function(value, spatial_shapes, sampling_locations,
                                     attention_weights) {
  n        <- value$size(1)
  n_heads  <- value$size(3)
  head_dim <- value$size(4)
  len_q    <- sampling_locations$size(2)
  n_levels <- sampling_locations$size(4)
  n_points <- sampling_locations$size(5)

  sizes <- as.integer(spatial_shapes[, 1] * spatial_shapes[, 2])
  value_list <- value$split(sizes, dim = 2)
  sampling_grids <- 2 * sampling_locations - 1

  sampling_value_list <- list()
  for (lvl in seq_len(n_levels)) {
    h_l <- as.integer(spatial_shapes[lvl, 1])
    w_l <- as.integer(spatial_shapes[lvl, 2])
    value_l <- value_list[[lvl]]$flatten(start_dim = 3)$transpose(2, 3)$
      reshape(c(n * n_heads, head_dim, h_l, w_l))
    grid_l <- sampling_grids[, , , lvl, , ]$transpose(2, 3)$
      flatten(start_dim = 1, end_dim = 2)  # (n*n_heads, len_q, n_points, 2)
    sampling_value_list[[lvl]] <- nnf_grid_sample(
      value_l, grid_l,
      mode = "bilinear", padding_mode = "zeros", align_corners = FALSE
    )  # (n*n_heads, head_dim, len_q, n_points)
  }

  # (n*n_heads, head_dim, len_q, n_levels, n_points) -> flatten levels*points
  sampled <- torch_stack(sampling_value_list, dim = -2)$
    flatten(start_dim = 4)
  attn <- attention_weights$transpose(2, 3)$
    reshape(c(n * n_heads, 1, len_q, n_levels * n_points))
  output <- (sampled * attn)$sum(-1)$
    view(c(n, n_heads * head_dim, len_q))
  output$transpose(2, 3)  # (n, len_q, n_heads*head_dim)
}

make_inputs <- function(device, requires_grad = FALSE) {
  torch_manual_seed(1)
  n <- 2L; n_heads <- 4L; head_dim <- 8L
  n_levels <- 2L; n_points <- 3L; len_q <- 5L
  shapes <- matrix(c(6L, 4L, 3L, 2L), ncol = 2, byrow = TRUE)
  spatial_shapes <- torch_tensor(shapes, dtype = torch_long())$to(device = device)
  level_start_index <- torch_cat(list(
    torch_zeros(1, dtype = torch_long()),
    (spatial_shapes[, 1] * spatial_shapes[, 2])$cumsum(1)[1:(n_levels - 1)]
  ))$to(device = device)
  len_in <- sum(as.integer(shapes[, 1] * shapes[, 2]))

  value <- torch_rand(n, len_in, n_heads, head_dim, device = device,
                      requires_grad = requires_grad)
  sampling_locations <- torch_rand(n, len_q, n_heads, n_levels, n_points, 2,
                                   device = device, requires_grad = requires_grad)
  attention_weights <- torch_rand(n, len_q, n_heads, n_levels, n_points,
                                  device = device, requires_grad = requires_grad)

  list(value = value, spatial_shapes = spatial_shapes,
       level_start_index = level_start_index,
       sampling_locations = sampling_locations,
       attention_weights = attention_weights)
}

test_that("ms_deform_attn matches the grid_sample reference (forward)", {
  skip_if_not(cuda_is_available())
  x <- make_inputs("cuda")

  out <- ops_ms_deform_attn(
    x$value, x$spatial_shapes, x$level_start_index,
    x$sampling_locations, x$attention_weights, 64L
  )
  ref <- ms_deform_attn_reference(
    x$value, x$spatial_shapes, x$sampling_locations, x$attention_weights
  )

  expect_equal(out$shape, ref$shape)
  expect_true(torch_allclose(out, ref, atol = 1e-4, rtol = 1e-4))
})

test_that("ms_deform_attn gradients match the reference (backward)", {
  skip_if_not(cuda_is_available())
  x <- make_inputs("cuda", requires_grad = TRUE)
  xr <- make_inputs("cuda", requires_grad = TRUE)

  ops_ms_deform_attn(
    x$value, x$spatial_shapes, x$level_start_index,
    x$sampling_locations, x$attention_weights, 64L
  )$sum()$backward()

  ms_deform_attn_reference(
    xr$value, xr$spatial_shapes, xr$sampling_locations, xr$attention_weights
  )$sum()$backward()

  expect_true(torch_allclose(x$value$grad, xr$value$grad, atol = 1e-3, rtol = 1e-3))
  expect_true(torch_allclose(x$sampling_locations$grad, xr$sampling_locations$grad,
                             atol = 1e-3, rtol = 1e-3))
  expect_true(torch_allclose(x$attention_weights$grad, xr$attention_weights$grad,
                             atol = 1e-3, rtol = 1e-3))
})

test_that("ms_deform_attn raises on CPU tensors", {
  x <- make_inputs("cpu")
  expect_error(
    ops_ms_deform_attn(
      x$value, x$spatial_shapes, x$level_start_index,
      x$sampling_locations, x$attention_weights, 64L
    ),
    regexp = "not implemented on the CPU"
  )
})
