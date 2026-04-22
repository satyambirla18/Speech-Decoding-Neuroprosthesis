from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import h5py
import numpy as np
import torch

def ascii_decoder(arr):
    if arr is None:
        return ""
    arr = np.asarray(arr).reshape(-1) #oneD array
    chararr = [chr(int(x)) for x in arr if int(x) !=0] #get actual transcript
    return "".join(chararr).strip() #leading and trailing spaces are useless here


def safe_int(x:Any, default:Optional[int]=None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default

@dataclass(frozen=True) #build this class init, repr and eq methods automatically
class TrialRef:
    file_path: str
    key: str
    session: str
    block_num: int
    trial_num: int
    n_time_steps: int
    seq_len: Optional[int] = None

from torch.utils.data import Dataset

# split can be in {"train", "val", "test"} sep logic for each
class BtTDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split = "train",
            target_kind = "phoneme",
            max_time_steps = None,
            transform = None,
            return_transcript = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.target_kind = target_kind
        self.max_time_steps = max_time_steps
        self.transform = transform
        self.return_transcript = return_transcript

        self._index = []
        self._h5_cache = {} #for multi workers in DataLoader, get_hf function doers this, each worker opens lazily and caches its file handles
        self._build_index() #populate _index by scanning files and creating TrialRef entries

    #need a function for multi worker process to serialize object and pass it to worker processes. Need handle for open h5py file that cannot be pickled. 
    #First copy all instance variables, then initialize _h5_cache as empty dict in worker process

    def __getstate__(self):
        state = self.__dict__.copy() #copy all instance variables
        state["_h5_cache"] = {} 
        return state #now we can do pickling in multi worker processes
    
    #trying to close all hdf5 files that were opened and cached
    def close(self):
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache = {}

    def _get_h5(self,file_path):
        if file_path in self._h5_cache:
            return self._h5_cache[file_path]
        # swmr=True is helpful for concurrent reads
        try:
            f = h5py.File(file_path, "r", swmr=True)
        except TypeError:
            f = h5py.File(file_path, "r")
        self._h5_cache[file_path] = f
        return f
    
    def _iter_split_files(self):
        suffix = f"data_{self.split}.hdf5"
        files = []
        for root, _, fnames in os.walk(self.root_dir):
            for fn in fnames:
                if fn.endswith(suffix):
                    files.append(os.path.join(root, fn))

        files.sort()
        # if len(files) == 0:
        #     raise RuntimeError(f"No data files found for split '{self.split}' in root dir '{self.root_dir}'")
        return files
    
    def _build_index(self) -> None:
        self._index.clear()
        h5_files = self._iter_split_files()

        for fp in h5_files:
            with h5py.File(fp, "r") as f:
                keys = sorted(list(f.keys()))
                for key in keys:
                    g = f[key]

                    session = str(g.attrs.get("session", ""))
                    block_num = safe_int(g.attrs.get("block_num", 0), 0) or 0
                    trial_num = safe_int(g.attrs.get("trial_num", 0), 0) or 0

                    n_time_steps = safe_int(g.attrs.get("n_time_steps", None), None)
                    # if n_time_steps is None:
                    #     n_time_steps = int(g["input_features"].shape[0])

                    seq_len = safe_int(g.attrs.get("seq_len", None), None)

                    self._index.append(
                        TrialRef(
                            file_path=fp,
                            key=key,
                            session=session,
                            block_num=block_num,
                            trial_num=trial_num,
                            n_time_steps=n_time_steps,
                            seq_len=seq_len,
                        )
                    )


    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref = self._index[idx]
        f = self._get_h5(ref.file_path)
        g = f[ref.key]

        x = np.asarray(g["input_features"], dtype=np.float32)

        if self.max_time_steps is not None and x.shape[0] > self.max_time_steps:
            x = x[: self.max_time_steps]

        if self.transform is not None:
            x = self.transform(x)

        out: Dict[str, Any] = {
            "inputs": torch.from_numpy(x),       
            "input_length": int(x.shape[0]),
            "session": ref.session,
            "block_num": ref.block_num,
            "trial_num": ref.trial_num,
            "file_path": ref.file_path,
            "key": ref.key,
        }

        # Labels exist only in train/val so separate handle
        if self.split != "test" and self.target_kind != "none":
            # seq_class_ids are padded with zeros, real length is seq_len
            if "seq_class_ids" in g:
                y = np.asarray(g["seq_class_ids"], dtype=np.int64).reshape(-1)
                if ref.seq_len is not None:
                    y = y[: ref.seq_len]
                else:
                    # fallback to strip padding zeros
                    y = y[y != 0]
            else:
                y = np.zeros((0,), dtype=np.int64)

            out["targets"] = torch.from_numpy(y)
            out["target_length"] = int(y.shape[0])

            if self.return_transcript and "transcription" in g:
                out["transcript"] = ascii_decoder(g["transcription"][...])
            else:
                out["transcript"] = ""

            if "sentence_label" in g.attrs:
                out["sentence_label"] = str(g.attrs["sentence_label"])

        return out

if __name__ == "__main__":
    root = "hdf5_data_final"
    ds = BtTDataset(root, split="train")
    print("num_samples:", len(ds))
    ex = ds[287]
    print("example keys:", list(ex.keys()))
    print("inputs:", ex["inputs"].shape)
    if "targets" in ex:
        print("targets:", ex["targets"].shape, "transcript:", ex.get("transcript", "")[:60])
