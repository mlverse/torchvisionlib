
test_that("We can load a detection model", {
  tmp <- tempfile(fileext = ".pt")
  url <- "https://storage.googleapis.com/torch-lantern-builds/testing-models/fasterrcnn_mobilenet_v3_large_320_fpn.pt"
  download.file(url, destfile = tmp, mode = "wb")
  model <- torch::jit_load(tmp)
  x <- list(torch_rand(3, 300, 400), torch_rand(3, 500, 400))
  predictions <- model(x)

  expect_length(predictions, 2)
  expect_length(predictions[[2]], 2)
  expect_length(predictions[[2]][[1]], 3)
  expect_length(predictions[[2]][[2]], 3)
})
