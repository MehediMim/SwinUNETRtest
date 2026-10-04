# Earlier validation

The following records checks from the original implementation environment.

- All Python files passed compileall syntax checks.
- Synthetic NIfTI test passed: nontrivial orientation, 3 mm slice spacing,
  resampling to 1 x 1 x 2 mm, binary-label preservation, padding/cropping,
  and export to the original shape/affine.
- Synthetic label round-trip Dice was 1.0000 for the test cuboid. This is a
  geometry check, not a model segmentation result.
- Model constructor and sliding-window arguments were checked against MONAI 0.9.1 documentation.
- Full model forward/backward and training were NOT run here. PyTorch 2.0.1
  could not be installed for the generation environment's Python version.
  The delivered setup explicitly uses Python 3.10. Run `python -m tests.smoke_test` in that
  environment before training.
- Real CT files, annotation completeness, HU calibration, CUDA availability,
  and memory fit were not verified because the user's Windows drive is unavailable.

## Project reorganization

- Source files compile after moving implementation into the `femur` package.
- Internal package imports resolve to existing modules.
- Default configuration loads from `configs/default.json`, including from a
  different working directory; configuration values were preserved.
- Patient pairing and split separation were checked with temporary file fixtures.
- Redundant root launchers and compatibility modules were removed; use the
  package commands documented in README.md.
- Model construction, preprocessing, training computations, checkpoint structure,
  and prediction computations were preserved.
- Full synthetic geometry/model checks and training were not rerun: the selected
  Python 3.10 environment lacks NumPy, NiBabel, SciPy, Matplotlib, PyTorch, and MONAI.
  Run `python -m tests.smoke_test --data-only` and then `python -m tests.smoke_test --device cuda`
  in the configured research environment.
