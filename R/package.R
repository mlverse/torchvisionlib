## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @importFrom utils download.file packageDescription unzip
## usethis namespace: end
NULL

.onLoad <- function(lib, pkg) {
  if (torch::torch_is_installed()) {

    if (!vision_is_installed())
      install_vision()

    if (!vision_is_installed()) {
      if (interactive())
        warning("libvision is not installed. Run `intall_vision()` before using the package.")
    } else {
      dyn.load(lib_path("torchvision"), local = FALSE)
      dyn.load(lib_path("vision"), local = FALSE)

      # when using devtools::load_all() the library might be available in
      # `lib/pkg/src`
      pkgload <- file.path(lib, pkg, "src", paste0(pkg, .Platform$dynlib.ext))
      if (file.exists(pkgload))
        dyn.load(pkgload)
      else
        library.dynam("torchvisionlib", pkg, lib)
    }
  }
}

inst_path <- function() {
  install_path <- Sys.getenv("VISION_HOME")
  if (nzchar(install_path)) return(install_path)

  system.file("", package = "torchvisionlib")
}

lib_path <- function(name = "vision") {
  install_path <- inst_path()

  if (.Platform$OS.type == "unix") {
    file.path(install_path, "lib", paste0("lib", name, lib_ext()))
  } else {
    file.path(install_path, "lib", paste0(name, lib_ext()))
  }
}

lib_ext <- function() {
  if (grepl("darwin", version$os))
    ".dylib"
  else if (grepl("linux", version$os))
    ".so"
  else
    ".dll"
}

vision_is_installed <- function() {
  file.exists(lib_path())
}

install_vision <- function(url = Sys.getenv("VISION_URL", unset = NA)) {

  if (!interactive() && Sys.getenv("TORCH_INSTALL", unset = 0) == "0") return()

  if (is.na(url)) {
    tmp <- tempfile(fileext = ".zip")
    version <- packageDescription("vision")$Version
    os <- get_cmake_style_os()
    dev <- if (torch::cuda_is_available()) "cu" else "cpu"

    url <- sprintf("https://github.com/mlverse/torchvisionlib/releases/download/libvision/vision-%s+%s-%s.zip",
                   version, dev, os)
  }

  if (is_url(url)) {
    file <- tempfile(fileext = ".zip")
    on.exit(unlink(file), add = TRUE)
    download.file(url = url, destfile = file)
  } else {
    message('Using file ', url)
    file <- url
  }

  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  unzip(file, exdir = tmp)

  file.copy(
    list.files(list.files(tmp, full.names = TRUE), full.names = TRUE),
    inst_path(),
    recursive = TRUE
  )
}

get_cmake_style_os <- function() {
  os <- version$os
  if (grepl("darwin", os)) {
    "Darwin"
  } else if (grepl("linux", os)) {
    "Linux"
  } else {
    "win64"
  }
}

is_url <- function(x) {
  grepl("^https", x) || grepl("^http", x)
}

