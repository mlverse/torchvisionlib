## usethis namespace: start
#' @importFrom Rcpp sourceCpp
#' @importFrom utils download.file packageDescription unzip
## usethis namespace: end
NULL

.onLoad <- function(lib, pkg) {
  if (torch::torch_is_installed()) {

    if (!torchvisionlib_is_installed())
      install_torchvisionlib()

    if (!torchvisionlib_is_installed()) {
      if (interactive())
        warning("torchvisionlib is not installed. Run `intall_torchvisionlib()` before using the package.")
    } else {
      if (grepl("mingw", R.version[["os"]])) {
        libpath <- lib_path("torchvisionlib")
        withr::with_dir(dirname(libpath), {
          dyn.load(basename(libpath), local = FALSE)
        })
      } else {
        dyn.load(lib_path("torchvision"), local = FALSE)
        dyn.load(lib_path("torchvisionlib"), local = FALSE)
      }

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
  install_path <- Sys.getenv("TORCHVISIONLIB_HOME")
  if (nzchar(install_path)) return(install_path)

  system.file("", package = "torchvisionlib")
}

lib_path <- function(name = "torchvisionlib") {
  install_path <- inst_path()

  if (.Platform$OS.type == "unix") {
    if (file.exists(file.path(install_path, "lib64"))) {
      file.path(install_path, "lib64", paste0("lib", name, lib_ext()))
    } else {
      file.path(install_path, "lib", paste0("lib", name, lib_ext()))
    }
  } else {
    file.path(install_path, "bin", paste0(name, lib_ext()))
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

#' Checks if an installation of torchvisionlib was found.
#' @rdname install_torchvisionlib
#' @export
torchvisionlib_is_installed <- function() {
  file.exists(lib_path())
}

#' Install additional libraries
#'
#' @param url Url for the binaries. Can also be the file path to the binaries.
#'
#' @export
install_torchvisionlib <- function(url = Sys.getenv("TORCHVISIONLIB_URL", unset = NA)) {

  if (!interactive() && Sys.getenv("TORCH_INSTALL", unset = 0) == "0") return()

  if (is.na(url)) {
    tmp <- tempfile(fileext = ".zip")
    version <- packageDescription("torchvisionlib")$Version
    os <- get_cmake_style_os()
    dev <- if (torch::cuda_is_available()) "cu" else "cpu"

    if (grepl("darwin", R.version$os)) {
      if (grepl("aarch64", R.version$arch)) {
        dev <- paste0(dev, "+arch64")
      } else {
        dev <- paste0(dev, "+x86_64")
      }
    }

    if (dev == "cu") {
      runtime_version <- torch::cuda_runtime_version()
      dev <- paste0(dev, runtime_version[1,1], runtime_version[1,2])
    }
    url <- sprintf("https://github.com/mlverse/torchvisionlib/releases/download/v%s/torchvisionlib-%s+%s-%s.zip",
                   version, version, dev, os)
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

