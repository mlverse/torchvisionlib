#include <Rcpp.h>
#include <iostream>
#define TORCHVISIONLIB_HEADERS_ONLY
#include <torchvisionlib/torchvisionlib.h>
#define TORCH_IMPL
#define IMPORT_TORCH
#include <torch.h>

void host_exception_handler ()
{
  if (torchvisionlib_last_error())
  {
    auto msg = Rcpp::as<std::string>(torch::string(torchvisionlib_last_error()));
    torchvisionlib_last_error_clear();
    Rcpp::stop(msg);
  }
}

// [[Rcpp::export]]
int test_f (torch::string path)
{
  return _test(path.get());
}
