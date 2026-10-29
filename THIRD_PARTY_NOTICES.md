# Third-party notices

- DINOv2: Meta Platforms, Inc. and affiliates; upstream source:
  https://github.com/facebookresearch/dinov2 . A subset of the inference code
  from the experiment snapshot is included under Apache-2.0. Original source
  headers and `third_party/dinov2/LICENSE` are retained. `hubconf.py` is narrowed
  to the required constructor. Official pretrained weights are downloaded separately and retain upstream terms.
- CBAM operations are a project implementation following the published channel
  and spatial attention design and the authors' reference:
  https://github.com/Jongchan/attention-module . This release copies the project's
  supplementary CBAM implementation, not the complete upstream repository.
- CECF uses cosine attention scaling and depthwise convolution concepts discussed
  in Swin Transformer V2 and CvT. These operators are adaptations for this task.
- Runtime dependencies (PyTorch, NumPy, OpenCV, SciPy, scikit-learn, pytest)
  retain their respective licenses and are installed separately.
- AeBAD-S and KSDD2 images and annotations are not distributed by this repository.
  Obtain them from the dataset authors and follow their terms.
