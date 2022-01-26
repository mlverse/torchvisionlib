
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
#' ops_nms(torch::torch_rand(3, 4), torch::torch_rand(3), 0.5)
#' }
#'
#' @family ops
#' @export
ops_nms <- function(dets, scores, iou_threshold) {
  rcpp_vision_ops_nms(dets, scores, iou_threshold)$add(1L)
}
