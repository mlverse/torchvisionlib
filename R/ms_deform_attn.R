#' Multiscale Deformable Attention
#'
#' Applies multiscale deformable attention to input feature maps.
#'
#' @param value (N, sum(H_l * W_l), C) input tensor
#' @param spatial_shapes (L, 2) tensor of feature map heights and widths
#' @param level_start_index (L,) long tensor, starting index for each level
#' @param sampling_loc (N, Lq, L, P, 2) normalized [0,1] sampling coordinates
#' @param attn_weight (N, Lq, L, P) attention weights
#' @param im2col_step (int) number of samples processed per kernel launch (tune for memory)
#'
#' @return (N, Lq, C) attended output tensor
#'
#' @details
#' This function is fully differentiable. Gradients are computed using
#' custom CUDA kernels.
#'
#' @examples
#' if (torch::torch_cuda_is_available()) {
#'   value <- torch::torch_randn(1, 100, 256)$to(device = "cuda")
#'   spatial_shapes <- torch::torch_tensor(matrix(c(10,10, 5,5), ncol=2, byrow=TRUE), dtype = torch::torch_long())$to(device = "cuda")
#'   level_start_index <- torch::torch_tensor(c(0, 100), dtype = torch::torch_long())$to(device = "cuda")
#'   sampling_loc <- torch::torch_rand(1, 10, 2, 4, 2)$to(device = "cuda")
#'   attn_weight <- torch::torch_rand(1, 10, 2, 4)$to(device = "cuda") %>% torch::torch_softmax(dim=3)
#'
#'   out <- nnf_multiscale_deformable_attn(
#'     value, spatial_shapes, level_start_index,
#'     sampling_loc, attn_weight
#'   )
#'   print(out$shape)
#' }
#'
#' @export
nnf_multiscale_deformable_attn <- function(
    value,
    spatial_shapes,
    level_start_index,
    sampling_loc,
    attn_weight,
    im2col_step = 64L
) {
  # Device check
  device <- value$device
  stopifnot("All tensors must be on the same device" = all(
      spatial_shapes$device == device,
      level_start_index$device == device,
      sampling_loc$device == device,
      attn_weight$device == device
    ))


  # Call exported autograd-enabled function
  out <- .Call(
    "multiscale_deformable_attn",
    value,
    spatial_shapes,
    level_start_index,
    sampling_loc,
    attn_weight,
    as.integer(im2col_step)
  )

  torch::torch_tensor(out, device = device)
}

#' Multiscale Deformable Attention Module
#'
#' @param d_model Dimension of model features
#' @param n_levels Number of feature levels
#' @param n_heads Number of attention heads
#' @param n_points Number of sampling points per level
#'
#' @return A multiscale deformable attention module
#' @export
nn_ms_deform_attn <- torch::nn_module(
  "nn_ms_deform_attn",

  initialize = function(d_model, n_levels, n_heads, n_points) {
    self$d_model <- d_model
    self$n_levels <- n_levels
    self$n_heads <- n_heads
    self$n_points <- n_points

    # Sampling offsets - learnable parameters for sampling locations
    self$sampling_offsets <- torch::nn_linear(d_model, n_heads * n_levels * n_points * 2)

    # Attention weights - learnable parameters for attention weights
    self$attention_weights <- torch::nn_linear(d_model, n_heads * n_levels * n_points)

    # Output projection
    self$output_proj <- torch::nn_linear(d_model, d_model)

    # Initialize weights
    self$sampling_offsets$weight <- torch::nn_init_xavier_uniform_(self$sampling_offsets$weight)
    self$sampling_offsets$bias$zero_()

    self$attention_weights$weight <- torch::nn_init_xavier_uniform_(self$attention_weights$weight)
    self$attention_weights$bias$zero_()

    self$output_proj$weight <- torch::nn_init_xavier_uniform_(self$output_proj$weight)
    self$output_proj$bias$zero_()
  },

  forward = function(query, reference_points, input_flatten, input_spatial_shapes,
                     input_level_start_index, input_padding_mask = NULL) {

    N <- query$size(1)  # batch size
    Lq <- query$size(2)  # query length
    L <- input_spatial_shapes$size(1)  # number of levels
    P <- self$n_points  # number of points

    # Compute sampling locations
    sampling_offsets <- self$sampling_offsets(query)
    sampling_offsets <- sampling_offsets$view(c(N, Lq, self$n_heads, L, P, 2))

    # Compute attention weights
    attention_weights <- self$attention_weights(query)
    attention_weights <- attention_weights$view(c(N, Lq, self$n_heads, L, P))
    attention_weights <- torch::torch_softmax(attention_weights, dim = -1)

    # Reshape reference points to match sampling offsets
    reference_points <- reference_points$unsqueeze(-1)$unsqueeze(-1)  # (N, Lq, 1, 1, 1, 2)
    reference_points <- reference_points$expand(c(N, Lq, self$n_heads, L, P, 2))

    # Combine reference points with sampling offsets
    sampling_locations <- reference_points + sampling_offsets

    # Reshape for the functional call
    sampling_locations <- sampling_locations$view(c(N, Lq, L * P * 2))
    attention_weights <- attention_weights$view(c(N, Lq, self$n_heads * L * P))

    # Apply attention function
    output <- nnf_multiscale_deformable_attn(
      value = input_flatten,
      spatial_shapes = input_spatial_shapes,
      level_start_index = input_level_start_index,
      sampling_loc = sampling_locations,
      attn_weight = attention_weights
    )

    # Project output
    output <- self$output_proj(output)

    return(output)
  }
)
