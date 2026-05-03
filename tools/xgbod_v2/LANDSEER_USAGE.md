# pre_xgbod Landseer Usage

This container runs pre-stage outlier filtering and rewrites test-set artifacts.

## Input artifact contract

Preferred core files in `--input-dir`:
- `data.npy`
- `labels.npy`
- `test_data.npy`
- `test_labels.npy`


## Output behavior

Writes:
- `data.npy`, `labels.npy` (train set pass-through)
- `test_data.npy`, `test_labels.npy` (test set filtered by XGBOD prediction)
- `xgbod_out.npy` (predicted anomaly labels for test set)


