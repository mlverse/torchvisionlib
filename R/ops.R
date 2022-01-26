
#' Performs non-maximum suppression (NMS) on the boxes
#'
#' Performs non-maximum suppression (NMS) on the boxes according to their
#' intersection-over-union (IoU).
#'
#' @details
#' NMS iteratively removes lower scoring boxes which have an IoU greater than
#' `iou_threshold` with another (higher scoring) box.
#'
#' If multiple boxes have the exact same score and satisfy the IoU criterion with
#' respect to a reference box, the selected box is not guaranteed to be the same
#' between CPU and GPU. This is similar to the behavior of argsort in PyTorch
#' when repeated values are present.
#'
#' @param boxes `Tensor[N,4]` boxes to perform NMS on. They are expected to be
#'  in `(x1, y1, x2, y2)` format with `0 <= x1 < x2` and `0 <= y1 < y2`.
#' @param scores `Tensor[N]` scores for each one of the boxes.
#' @param iou_threshold `float` discards all overlapping boxes with `IoU > iou_threshold`.
#'
#' @returns
#' int64 tensor with the indices of the elements that have been kept by NMS,
#' sorted in decreasing order of scores
#'
#' @examples
#' if (torchvisionlib_is_installed()) {
#'   ops_nms(torch::torch_rand(3, 4), torch::torch_rand(3), 0.5)
#' }
#' @family ops
#' @export
ops_nms <- function(dets, scores, iou_threshold) {
  rcpp_vision_ops_nms(dets, scores, iou_threshold)$add(1L)
}


#' Performs Deformable Convolution v2,
#'
#' Ddescribed in [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)
#' if `mask` is not `NULL` and performs Deformable Convolution, described in
#' [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)
#' if `mask` is `NULL`.
#'
#' @param input (`Tensor[batch_size, in_channels, in_height, in_width]`): input tensor
#' @param offset (`Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]`):
#'   offsets to be applied for each position in the convolution kernel.
#' @param weight (`Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]`): convolution weights,
#'  split into groups of size (in_channels // groups)
#' @param bias (`Tensor[out_channels]`): optional bias of shape (out_channels,). Default: `NULL`
#' @param stride (int or `Tuple[int, int]`): distance between convolution centers. Default: 1
#' @param padding (int or `Tuple[int, int]`): height/width of padding of zeroes around
#'  each image. Default: 0
#' @param dilation (int or `Tuple[int, int]`): the spacing between kernel elements. Default: 1
#' @param mask (`Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]`):
#'   masks to be applied for each position in the convolution kernel. Default: `NULL`
#'
#' @returns
#'   `Tensor[batch_sz, out_channels, out_h, out_w]`: result of convolution
#'
#' @examples
#' if (torchvisionlib_is_installed()) {
#'   library(torch)
#'   input <- torch_rand(4, 3, 10, 10)
#'   kh <- kw <- 3
#'   weight <- torch_rand(5, 3, kh, kw)
#'   # offset and mask should have the same spatial size as the output
#'   # of the convolution. In this case, for an input of 10, stride of 1
#'   # and kernel size of 3, without padding, the output size is 8
#'   offset <- torch_rand(4, 2 * kh * kw, 8, 8)
#'   mask <- torch_rand(4, kh * kw, 8, 8)
#'   out <- ops_deform_conv2d(input, offset, weight, mask = mask)
#'   print(out$shape)
#' }
#' @export
ops_deform_conv2d <- function(input,
                              offset,
                              weight,
                              bias = NULL,
                              stride = c(1, 1),
                              padding = c(0, 0),
                              dilation = c(1, 1),
                              mask = NULL) {

  out_channels <- weight$shape[1]
  use_mask <- !is.null(mask)

  if (is.null(mask)) {
    mask <- torch::torch_zeros(input$shape[1], 0, device=input$device, dtype=input$dtype)
  }

  if (is.null(bias)) {
    bias = torch::torch_zeros(out_channels, device=input$device, dtype=input$dtype)
  }

  strides <- .pair(stride)
  pads <- .pair(padding)
  dils <- .pair(dilation)
  weights <- tail(weight$shape, 2)
  n_in_channels <- input$shape[2]

  n_offset_grps <- offset$shape[2] %/% (2 * weights[1] * weights[2])
  n_weight_grps <- n_in_channels %/% weight$shape[2]

  if (n_offset_grps == 0) {
    runtime_error(glue::glue(
      "the shape of the offset tensor at dimension 1 is not valid. It should ",
      "be a multiple of 2 * weight$size[3] * weight$size[4].\n",
      "Got offset$shape[2]={offset$shape[2]}, while 2 * weight$size[3] * weight$size[4]={2 * weights[1] * weights[2]}"
    ))
  }

  rcpp_vision_ops_deform_conv2d(
    input,
    weight,
    offset,
    mask,
    bias,
    strides[1],
    strides[2],
    pads[1],
    pads[2],
    dils[1],
    dils[2],
    n_weight_grps,
    n_offset_grps,
    use_mask
  )
}
