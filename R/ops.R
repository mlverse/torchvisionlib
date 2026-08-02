
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
ops_nms <- function(boxes, scores, iou_threshold) {
  rcpp_vision_ops_nms(boxes, scores, iou_threshold)$add(1L)
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
  weights <- utils::tail(weight$shape, 2)
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

#' Multi-scale deformable attention
#'
#' Computes multi-scale deformable attention as used by Deformable-DETR and
#' LW-DETR. Only a CUDA implementation is provided; all input tensors must be on
#' the same CUDA device. On CPU, models should fall back to a pure-R
#' implementation (e.g. based on [torch::nnf_grid_sample()]).
#'
#' @details
#' `spatial_shapes` and `level_start_index` use the kernel's 0-based indexing
#' convention: `level_start_index[l]` is the flat offset (into the `Len_in`
#' dimension of `value`) of the first element of level `l`, and the output is a
#' feature tensor, so no 1-based index adjustment is applied.
#'
#' @param value (`Tensor[batch, Len_in, n_heads, head_dim]`): flattened
#'   multi-scale feature values.
#' @param spatial_shapes (`Tensor[n_levels, 2]`, integer): the `(H, W)` of each
#'   feature level. The sum of `H * W` over levels must equal `Len_in`.
#' @param level_start_index (`Tensor[n_levels]`, integer): 0-based start offset
#'   of each level within `Len_in`.
#' @param sampling_locations (`Tensor[batch, Len_q, n_heads, n_levels, n_points, 2]`):
#'   sampling locations in `[0, 1]` (normalized `x, y`).
#' @param attention_weights (`Tensor[batch, Len_q, n_heads, n_levels, n_points]`):
#'   attention weights, typically normalized over the `n_levels * n_points` axis.
#' @param im2col_step (int): batch chunk size used internally by the kernel.
#'   Must divide `batch`. Default: 64.
#'
#' @returns
#'   `Tensor[batch, Len_q, n_heads * head_dim]`: the attended output.
#'
#' @family ops
#' @export
ops_ms_deform_attn <- function(value, spatial_shapes, level_start_index,
                               sampling_locations, attention_weights,
                               im2col_step = 64L) {
  rcpp_vision_ops_ms_deform_attn(
    value,
    spatial_shapes,
    level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step
  )
}


#' Performs Position-Sensitive Region of Interest (RoI) Align operator
#'
#' The (RoI) Align operator is mentioned in [Light-Head R-CNN](https://arxiv.org/abs/1711.07264).
#'
#' @param input (`Tensor[N, C, H, W]`): The input tensor, i.e. a batch with `N` elements. Each element
#'   contains `C` feature maps of dimensions `H x W`.
#' @param boxes (`Tensor[K, 5]` or `List[Tensor[L, 4]]`): the box coordinates in (x1, y1, x2, y2)
#'   format where the regions will be taken from.
#'   The coordinate must satisfy `0 <= x1 < x2` and `0 <= y1 < y2`.
#'   If a single Tensor is passed, then the first column should
#'   contain the index of the corresponding element in the batch, i.e. a number in `[1, N]`.
#'   If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
#'   in the batch.
#' @param output_size (int or `Tuple[int, int]`): the size of the output (in bins or pixels) after the pooling
#'   is performed, as (height, width).
#' @param spatial_scale (float): a scaling factor that maps the box coordinates to
#'   the input coordinates. For example, if your boxes are defined on the scale
#'   of a 224x224 image and your input is a 112x112 feature map (resulting from a 0.5x scaling of
#'   the original image), you'll want to set this to 0.5. Default: 1.0
#' @param sampling_ratio (int): number of sampling points in the interpolation grid
#'   used to compute the output value of each pooled output bin. If > 0,
#'   then exactly `sampling_ratio x sampling_ratio` sampling points per bin are used. If
#'   <= 0, then an adaptive number of grid points are used (computed as
#'   `ceil(roi_width / output_width)`, and likewise for height). Default: -1
#' @returns
#'   `Tensor[K, C / (output_size[1] * output_size[2]), output_size[1], output_size[2]]`:
#'   The pooled RoIs
#'
#' @examples
#' if (torchvisionlib_is_installed()) {
#' library(torch)
#' library(torchvisionlib)
#' input <- torch_randn(1, 3, 28, 28)
#' boxes <- list(torch_tensor(matrix(c(1,1,5,5), ncol = 4)))
#' roi <- nn_ps_roi_align(output_size = c(1, 1))
#' roi(input, boxes)
#' }
#'
#' @export
ops_ps_roi_align <- function(input, boxes, output_size, spatial_scale= 1,
                             sampling_ratio = -1) {
  rois <- boxes
  output_size <- .pair(output_size)
  if (!inherits(rois, "torch_tensor")) {
    rois <- convert_boxes_to_roi_format(rois)
  }
  rois[,1] <- rois[,1]-1
  out <- rcpp_vision_ops_ps_roi_align(input, rois, spatial_scale, output_size[1],
                               output_size[2], sampling_ratio)
  out[[1]]
}


#' @describeIn ops_ps_roi_align The [torch::nn_module()] wrapper for [ops_ps_roi_align()].
#' @export
nn_ps_roi_align <- torch::nn_module(
  initialize = function(output_size, spatial_scale= 1, sampling_ratio = -1) {
    self$output_size <- output_size
    self$spatial_scale <- spatial_scale
    self$sampling_ratio <- sampling_ratio
  },
  forward = function(input, rois) {
    ops_ps_roi_align(input, rois, self$output_size, self$spatial_scale,
                     self$sampling_ratio)
  }
)


#' RoI align pooling for rotated proposals
#'
#' Performs RoI align pooling for rotated proposals, as implemented by the MMCV
#' `roi_align_rotated` operator
#' (see <https://mmcv.readthedocs.io/en/latest/deployment/mmcv_ops_definition.html#mmcvroialignrotated>).
#' Only a CPU implementation is provided.
#'
#' @param input (`Tensor[N, C, H, W]`): input feature map.
#' @param rois (`Tensor[K, 6]`): rotated boxes with columns
#'   `(batch_index, cx, cy, w, h, angle)`, where `batch_index` is a **0-based**
#'   index into the first dimension of `input`, `(cx, cy)` is the box center,
#'   `(w, h)` the box size and `angle` the rotation angle in radians
#'   (counterclockwise unless `clockwise = TRUE`).
#' @param output_size (int or `Tuple[int, int]`): the output size `(height, width)`
#'   after pooling.
#' @param spatial_scale (float): scaling factor mapping box coordinates to input
#'   coordinates. For example, if boxes are defined on a 224x224 image and
#'   `input` is a 112x112 feature map, set this to 0.5.
#' @param sampling_ratio (int): number of sampling points per output bin. If `<= 0`,
#'   an adaptive number of grid points is used (computed as `ceil(roi_size / output_size)`).
#'   Default: 0
#' @param aligned (bool): if `TRUE` (default), the aligned implementation is
#'   used (results are shifted by -0.5 before interpolation, matching
#'   detectron2). If `FALSE`, the legacy MMDetection implementation is used and
#'   boxes are clamped to a minimum size of 1.
#' @param clockwise (bool): if `TRUE`, the rotation angle is interpreted in a
#'   clockwise fashion in image space, otherwise it is counterclockwise.
#'   Default: `FALSE`
#'
#' @returns
#' `Tensor[K, C, output_size[1], output_size[2]]`: the pooled features, where
#' the `r`-th element corresponds to the `r`-th RoI in `rois`.
#'
#' @examples
#' if (torchvisionlib_is_installed()) {
#'   library(torch)
#'   input <- torch_randn(1, 3, 28, 28)
#'   # (batch_index, cx, cy, w, h, angle) with 0-based batch index
#'   rois <- torch_tensor(matrix(c(0, 14, 14, 10, 10, 0.5), ncol = 6))
#'   ops_roi_align_rotated(input, rois, output_size = c(5, 5),
#'                         spatial_scale = 1, sampling_ratio = 2)
#' }
#'
#' @family ops
#' @export
ops_roi_align_rotated <- function(input, rois, output_size, spatial_scale,
                                  sampling_ratio = 0, aligned = TRUE,
                                  clockwise = FALSE) {
  output_size <- .pair(output_size)
  rcpp_vision_ops_roi_align_rotated(
    input, rois,
    output_size[1], output_size[2],
    spatial_scale, sampling_ratio,
    aligned, clockwise
  )
}


#' @describeIn ops_roi_align_rotated The [torch::nn_module()] wrapper for [ops_roi_align_rotated()].
#' @export
nn_roi_align_rotated <- torch::nn_module(
  initialize = function(output_size, spatial_scale, sampling_ratio = 0,
                        aligned = TRUE, clockwise = FALSE) {
    self$output_size <- output_size
    self$spatial_scale <- spatial_scale
    self$sampling_ratio <- sampling_ratio
    self$aligned <- aligned
    self$clockwise <- clockwise
  },
  forward = function(input, rois) {
    ops_roi_align_rotated(input, rois, self$output_size, self$spatial_scale,
                          self$sampling_ratio, self$aligned, self$clockwise)
  }
)


