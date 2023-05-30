#' Read JPEG's directly into torch tensors
#' @param path path to JPEG file
#'
#' @export
vision_read_jpeg <- function(path) {
  rcpp_vision_read_jpeg_float(path.expand(path))
}
