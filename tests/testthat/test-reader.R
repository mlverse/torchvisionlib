test_that("can read img", {
  img <- vision_read_jpeg(test_path("imgs/image_00001.jpg"))
  expect_tensor(img)
  expect_equal(img$dtype$.type(), "Float")
})
