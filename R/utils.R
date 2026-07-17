.pair <- function(x) {
  if (length(x) > 2 || length(x) < 1) {
    runtime_error(sprintf("Expected 1 or 2 values, got %d", length(x)))
  }
  if (length(x) == 1)
    c(x, x)
  else
    x
}

convert_boxes_to_roi_format <- function(boxes) {
  if (inherits(boxes, "torch_tensor")) boxes <- list(boxes)
  concat_boxes <- .cat(boxes, dim = 1)
  ids <- .cat(imap(boxes, function(b, i) {
    torch::torch_full_like(b[,1,drop=FALSE], i)
  }))
  torch::torch_cat(list(ids, concat_boxes), dim = 2)
}

runtime_error <- function(...) {
  rlang::abort(..., class = "runtime_error")
}

# Convert a set of rotated boxes to `cxcywhr`, the format expected by the
# `box_iou_rotated` C++ op. Conversions mirror torchvision's `box_convert`.
.rotated_boxes_to_cxcywhr <- function(boxes, fmt) {
  switch(
    fmt,
    cxcywhr = boxes,
    xywhr = .box_xywhr_to_cxcywhr(boxes),
    xyxyxyxy = .box_xyxyxyxy_to_cxcywhr(boxes),
    runtime_error(sprintf(
      "Unsupported format '%s'. Supported rotated formats: cxcywhr, xywhr, xyxyxyxy.",
      fmt
    ))
  )
}

.box_xywhr_to_cxcywhr <- function(boxes) {
  b <- torch::torch_unbind(boxes, dim = -1)
  x1 <- b[[1]]
  y1 <- b[[2]]
  w <- b[[3]]
  h <- b[[4]]
  r <- b[[5]]
  r_rad <- r * pi / 180
  cos <- torch::torch_cos(r_rad)
  sin <- torch::torch_sin(r_rad)
  cx <- x1 + w / 2 * cos + h / 2 * sin
  cy <- y1 - w / 2 * sin + h / 2 * cos
  torch::torch_stack(list(cx, cy, w, h, r), dim = -1)
}

.box_xyxyxyxy_to_cxcywhr <- function(boxes) {
  b <- torch::torch_unbind(boxes, dim = -1)
  x1 <- b[[1]]
  y1 <- b[[2]]
  x2 <- b[[3]]
  y2 <- b[[4]]
  x3 <- b[[5]]
  y3 <- b[[6]]
  r <- torch::torch_atan2(y1 - y2, x2 - x1) * 180 / pi
  w <- ((x2 - x1) * (x2 - x1) + (y1 - y2) * (y1 - y2))$sqrt()
  h <- ((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2))$sqrt()
  .box_xywhr_to_cxcywhr(torch::torch_stack(list(x1, y1, w, h, r), dim = -1))
}

# Efficient version of torch.cat that avoids a copy if there is only a single element in a list
.cat <- function(tensors, dim = 1) {
  if (length(tensors) == 1)
    return(tensors[[1]])

  torch::torch_cat(tensors, dim)
}

# base R implementation of purr map2 (copied from rlang.)
map2 <- function (.x, .y, .f, ...) {
  .f <- rlang::as_function(.f, env = rlang::global_env())
  out <- mapply(.f, .x, .y, MoreArgs = list(...), SIMPLIFY = FALSE)
  if (length(out) == length(.x)) {
    rlang::set_names(out, names(.x))
  }
  else {
    rlang::set_names(out, NULL)
  }

}
#' @importFrom rlang %||%
imap <- function (.x, .f, ...) {
  map2(.x, names(.x) %||% seq_along(.x), .f, ...)
}


