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
#'   out <- multiscale_deformable_attn(
#'     value, spatial_shapes, level_start_index,
#'     sampling_loc, attn_weight
#'   )
#'   print(out$shape)
#' }
#'
#' @export
multiscale_deformable_attn <- function(
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
