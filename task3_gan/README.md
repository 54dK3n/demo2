# Task 3: 64 x 64 OASIS DCGAN

The model trains only on the original grayscale MRI slices in
`keras_png_slices_train`. It does not read any `seg` directory.

From the repository root:

```bash
export OASIS_ROOT=/home/groups/comp3710/OASIS
python task3_gan/dataset.py \
  --image-dir "$OASIS_ROOT/keras_png_slices_train"
python task3_gan/train.py \
  --image-dir "$OASIS_ROOT/keras_png_slices_train" \
  --epochs 50 \
  --run-name dcgan_64_d_train_mode
```

For a short end-to-end check before a full cluster run:

```bash
python task3_gan/train.py --epochs 1 --max-batches 2 --num-workers 0 \
  --run-name smoke_run
```

Each run saves its configuration, real-image reference grid, an epoch-zero sample,
per-epoch fixed-noise sample grids, CSV metrics, generator/discriminator loss plots,
a latest checkpoint, and numbered periodic checkpoints under
`task3_gan/results/<run-name>/`.

The CSV includes discriminator confidence and simple sample-diversity indicators.
`near_duplicate_rate` near 1 or very low `sample_pixel_std` is a warning to inspect
the fixed-noise grids for mode collapse; these metrics are diagnostics, not a
substitute for visual review.

## Selected result

`dcgan_64_d_train_mode` is the result used for the course demonstration. At epoch
50 it recorded `sample_pixel_std=0.0940`, `mean_nearest_rmse=0.1175`, and a
`near_duplicate_rate` of `0.0%`. Its fixed-noise grid contains varied brain shapes.

The older `dcgan_64_baseline` result is not the final model: it ended with a
`near_duplicate_rate` of `60.9%` and visibly repeated samples. `smoke_run` is only
an end-to-end code check. Both remain local and are ignored by Git.

The root [README](../README.md#dcgan-brain-generation) contains the complete
metrics table and representative result images.
