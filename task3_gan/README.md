# Task 3: 64×64 OASIS DCGAN baseline

This baseline trains only on the original grayscale MRI slices in
`keras_png_slices_train`. It does not read any `seg` directory.

From the repository root:

```bash
python task3_gan/dataset.py
python task3_gan/train.py --run-name dcgan_64_baseline
```

For a short end-to-end check before a full cluster run:

```bash
python task3_gan/train.py --epochs 1 --max-batches 2 --num-workers 0 \
  --run-name smoke_run
```

Each run saves its configuration, real-image reference grid, an epoch-zero sample,
per-epoch fixed-noise sample grids, CSV metrics, loss plot, a latest checkpoint,
and numbered periodic checkpoints under `task3_gan/results/<run-name>/`.

The CSV includes discriminator confidence and simple sample-diversity indicators.
`near_duplicate_rate` near 1 or very low `sample_pixel_std` is a warning to inspect
the fixed-noise grids for mode collapse; these metrics are diagnostics, not a
substitute for visual review.
