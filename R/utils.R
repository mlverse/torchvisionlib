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


