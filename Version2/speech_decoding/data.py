from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from .utils import safe_int, decode_ascii_array

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

@dataclass(frozen=True)
class TrialRef:
    h5_path: str
    key: str

def _pick_key(grp: h5py.Group, candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in grp:
            return k
    return None

def _ensure_time_major(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={x.shape}")
    if x.shape[0] == 512 and x.shape[1] != 512:
        return x.T
    if x.shape[1] == 512:
        return x
    return x

class HDF5SpeechDataset(Dataset):

    def __init__(
        self,
        root: str,
        partition: Optional[str] = None,
        normalize: bool = True,
        clip_value: float = 5.0,
        max_trials: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.partition = partition
        self.normalize = normalize
        self.clip_value = clip_value

        if not self.root.exists():
            raise FileNotFoundError(f"Data root not found: {self.root}")

        self.trials: List[TrialRef] = self._build_index(max_trials=max_trials)

        if len(self.trials) == 0:
            raise RuntimeError(
                "No trials were indexed. Check your root path and HDF5 structure. "
                "Tip: ensure your folder contains *.hdf5 or data_{train,val,test}.hdf5."
            )

    def _build_index(self, max_trials: Optional[int]) -> List[TrialRef]:
        trials: List[TrialRef] = []

        if self.partition in {"train", "val", "test"}:
            candidate = self.root / f"data_{self.partition}.hdf5"
            if candidate.exists():
                with h5py.File(candidate, "r") as f:
                    for key in f.keys():
                        if isinstance(f[key], h5py.Group):
                            trials.append(TrialRef(str(candidate), key))
                            if max_trials and len(trials) >= max_trials:
                                return trials
                return trials

        for h5_path in sorted(self.root.rglob("*.hdf5")):
            try:
                with h5py.File(h5_path, "r") as f:
                    for key in f.keys():
                        if not isinstance(f[key], h5py.Group):
                            continue
                        grp = f[key]
                        feat_key = _pick_key(grp, ["neural_features", "input_features", "features", "x"])
                        if feat_key is None:
                            continue
                        trials.append(TrialRef(str(h5_path), key))
                        if max_trials and len(trials) >= max_trials:
                            return trials
            except OSError:
                continue

        return trials

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref = self.trials[idx]
        with h5py.File(ref.h5_path, "r", libver="latest") as f:
            grp = f[ref.key]

            feat_key = _pick_key(grp, ["neural_features", "input_features", "features", "x"])
            if feat_key is None:
                raise KeyError(f"Feature dataset not found in {ref.h5_path}:{ref.key}")

            x = grp[feat_key][()]
            x = np.asarray(x)
            x = _ensure_time_major(x)
            t = x.shape[0]
            n_time_steps = safe_int(grp["n_time_steps"][()]) if "n_time_steps" in grp else t

            seq_key = _pick_key(grp, ["seq_class_ids", "phoneme_ids", "y"])
            seq = grp[seq_key][()] if seq_key is not None else None
            if seq is not None:
                seq = np.asarray(seq).astype(np.int64)
                if "seq_len" in grp:
                    L = safe_int(grp["seq_len"][()])
                    seq = seq[:L]
            seq_len = int(len(seq)) if seq is not None else 0

            tr_key = _pick_key(grp, ["transcriptions", "transcription", "sentence_label"])
            transcription = ""
            if tr_key is not None:
                raw = grp[tr_key][()]
                transcription = decode_ascii_array(raw)

            xt = torch.from_numpy(x).float()
            if self.normalize:
                mean = xt.mean(dim=0, keepdim=True)
                std = xt.std(dim=0, keepdim=True).clamp_min(1e-5)
                xt = (xt - mean) / std
                if self.clip_value is not None and self.clip_value > 0:
                    xt = xt.clamp(-self.clip_value, self.clip_value)

            out: Dict[str, Any] = {
                "x": xt,
                "x_len": int(n_time_steps),
                "seq": torch.from_numpy(seq) if seq is not None else None,
                "seq_len": int(seq_len),
                "transcription": transcription,
                "ref": ref,
            }
            return out

def collate_ctc(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    x_lens = torch.tensor([b["x_len"] for b in batch], dtype=torch.long)
    T = int(x_lens.max().item())
    C = int(batch[0]["x"].shape[1])

    x = torch.zeros((B, T, C), dtype=torch.float32)
    for i, b in enumerate(batch):
        t = b["x"].shape[0]
        x[i, :t] = b["x"]

    target_lens = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    targets_list = []
    for b in batch:
        if b["seq"] is None:
            targets_list.append(torch.empty((0,), dtype=torch.long))
        else:
            targets_list.append(b["seq"].long())
    targets = torch.cat(targets_list, dim=0) if targets_list else torch.empty((0,), dtype=torch.long)

    return {
        "x": x,
        "x_len": x_lens,
        "targets": targets,
        "target_len": target_lens,
        "transcription": [b.get("transcription","") for b in batch],
        "ref": [b.get("ref") for b in batch],
    }
