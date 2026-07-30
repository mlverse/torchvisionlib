# torchvisionlib (development version)

- Added `ops_box_iou_rotated()`, a CPU implementation of intersection-over-union
  between rotated boxes, supporting the `cxcywhr`, `xywhr` and `xyxyxyxy`
  formats. Adapted from Detectron2 (Apache-2.0). (#31)
  
# torchvisionlib 0.8.0

- Updates to support LibTorch v2.8


# torchvisionlib 0.5.0

- Updates to support LibTorch v2.0.1

# torchvisionlib 0.4.0

- Updates to support LibTorch v1.13.1
- New faster image reader. By reading directly into a torch tensor it can be 2x times faster than `jpeg::readJPEG`. (#15)

# torchvisionlib 0.3.0

- Updates for LibTorch v1.12.1 (#9)
- Support for Apple Silicon (#11)

# torchvisionlib 0.2.0

# torchvisionlib 0.1.0.9000

* Added a `NEWS.md` file to track changes to the package.
