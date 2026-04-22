# Speech Decoding – Training Skeleton (CTC + Conformer + RoPE)

This is a *clean* training script built from your notebook code:
- **Dataset loader** (HDF5, variable length, robust field-name detection)
- **Conformer encoder** (with RoPE attention)
- **CTC head** for phoneme sequence training (BLANK index 0)
- **Training + eval** (greedy decode + edit distance → phoneme error rate proxy)

## What I fixed / made robust vs the notebooks

1. **HDF5 field-name mismatch**: your notes mention `neural_features`, notebook used `input_features`. Loader now checks both (+ a few common fallbacks).
2. **Feature shape mismatch**: dataset sometimes stores `(512, T)` but models expect `(T, 512)`. Loader auto-transposes when it detects `512` as the first dim.
3. **Missing helpers**: `TrialRef`, `safe_int`, and the ASCII decoder are implemented in `speech_decoding/utils.py`.
4. **CTC-ready batching**: variable length padding + concatenated targets with `collate_ctc`.
5. **Optimizer**: the notebook test used `torch.optim.Adafactor` (not available in standard PyTorch). Training uses **AdamW** by default.

## Install

```bash
pip install -r requirements.txt
```

## Run (example)

If your folder contains `data_train.hdf5` and `data_val.hdf5`:

```bash
python train.py --data-root /path/to/data_root --out runs/exp1 --batch-size 8 --epochs 20
```

If your folder contains many session subfolders with `*.hdf5`, it also works:

```bash
python train.py --data-root /path/to/hdf5_root --out runs/exp1
```

## Notes

- `vocab_size` defaults to 42 (your mapping length).
- Edit distance metric is computed over decoded **phoneme id sequences** as a quick sanity check.
- If your `test` split has no labels, set `--val-partition none`.
