test_that("test", {
  tmp <- tempfile(fileext = ".pt")
  download.file(url = "https://storage.googleapis.com/torch-lantern-builds/testing-models/resnet18.pt", destfile = tmp, mode = "wb")
  expect_error(test_f(normalizePath(tmp)), regexp = NA)
})
