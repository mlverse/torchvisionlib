
<!-- README.md is generated from README.Rmd. Please edit that file -->

# torchvisionlib

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html)
[![R-CMD-check](https://github.com/mlverse/torchvisionlib/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/mlverse/torchvisionlib/actions/workflows/R-CMD-check.yaml)
[![CRAN
status](https://www.r-pkg.org/badges/version/torchvisionlib)](https://CRAN.R-project.org/package=torchvisionlib)
[![](https://cranlogs.r-pkg.org/badges/torchvisionlib)](https://cran.r-project.org/package=torchvisionlib)
[![Discord](https://img.shields.io/discord/837019024499277855?logo=discord)](https://discord.com/invite/s3D5cKhBkx)
<!-- badges: end -->

The goal of torchvisionlib is to provide access to C++ operations
implemented in [torchvision](https://github.com/pytorch/vision). It
provides plain R acesss to some of those C++ operations but, most
importantly it provides full support for JIT operators defined in
[torchvision](https://github.com/pytorch/vision), allowing us to load
‘scripted’ object detection and image segmentation models.

## Installation

torchvisionlib can be installed from CRAN with:

``` r
install.packages("torchvisionlib")
```

You can also install the development version of torchvisionlib from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("mlverse/torchvisionlib")
```

## Example

Suppose that we want to load an image detection model implemented in
torchvision. First, in Python, we can save JIT script and then save this
model:

``` python
import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()

jit_model = torch.jit.script(model)
torch.jit.save(jit_model, "fasterrcnn_mobilenet_v3_large_320_fpn.pt")
```

We can then load this model in R. Simply loading torchvisionlib will
register all JIT operators, and we can use `torch::jit_load()`.

``` r
library(torchvisionlib)
model <- torch::jit_load("fasterrcnn_mobilenet_v3_large_320_fpn.pt")
model
#> An `nn_module` containing 19,386,354 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • transform: <script_module> #0 parameters
#> • backbone: <script_module> #4,414,944 parameters
#> • rpn: <script_module> #609,355 parameters
#> • roi_heads: <script_module> #14,362,055 parameters
```

You can then use this model to make preditions or even fine tuning.
