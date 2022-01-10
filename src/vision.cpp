#include <Rcpp.h>
#include <iostream>
#define LLTM_HEADERS_ONLY
#include "vision/vision.h"
#define TORCH_IMPL
#define IMPORT_TORCH
#include <torch.h>

// [[Rcpp::export]]
int test_f ()
{
  return test();
}
