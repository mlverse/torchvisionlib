.pair <- function(x) {
  if (length(x) > 2 || length(x) < 1) {
    runtime_error(sprintf("Expected 1 or 2 values, got %d", length(x)))
  }
  if (length(x) == 1)
    c(x, x)
  else
    x
}

runtime_error <- function(...) {
  rlang::abort(..., class = "runtime_error")
}
