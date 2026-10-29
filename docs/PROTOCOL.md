# Implementation and protocol

The frozen ViT-S/14 uses blocks 6/9/12, individually normalized and concatenated
to 1152 channels, then shared 1x1 projection to 96 channels. Full-image and crop
tokens are restored to a 72x72 grid. Four 518x518 crops have origins (0,0),
(0,490), (490,0), and (490,490) on the 1008x1008 canvas. Overlaps are averaged.

CECF uses local queries and full-image keys/values, attention dimension 24 with
2 heads, parallel regular/shifted 6x6 windows, relative position bias, residual
depthwise convolution, and a tanh residual gate. The decoder produces a 288x288
map. Classification receives the normalized full-image CLS and the mean of the
largest 830 probabilities on the padded decoder grid. For the final map readout,
padding is removed first and the largest min(830, number of valid grid cells)
probabilities are averaged. This is a fixed count, including on KSDD2, rather
than a separately rounded 1% of each non-square grid.

Image readout combines probabilities in logit space at head weight 0.5 by default,
with probability clipping at 1e-5. The loss is positive-weighted (8) pixel BCE,
0.3 times Dice loss on anomalous images, and 0.2 times image BCE. Padded regions
are excluded from the supervised pixel loss. R5 removes the image term.

Model selection uses pooled development grid pixel AP. Masks are area-resized
to each prediction grid and binarized at >0.02. Calibration selects image F1
subject to recall >=0.90 and pixel threshold by anomalous-image mean Dice.
Final pixel metrics are computed in original image coordinates using exact cell
area counts, not equal weighting of differently sized images' grid pixels.

AeBAD-S uses a supervised split with shared development and calibration data.
Images are deduplicated by hash. KSDD2 uses disjoint fit, development, and
calibration partitions and the official test split. Each dataset is trained
separately.

Important source files:

- `src/bladegl/backbone.py`: views, frozen encoder, common projection/decoder.
- `src/bladegl/model.py`: CECF and full fusion head.
- `src/bladegl/variants.py`: original structural/training variants.
- `src/bladegl/cbam.py`: supplementary CBAM alternative.
- `src/bladegl/runner.py`: raw RGB training, selection, calibration, evaluation.
- `src/metrics.py`: original-pixel metrics and threshold rules.
- `src/protocol.py`: canonical manifests and disjoint-role checks.
