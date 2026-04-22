#need to build a batching algo to handle variable length rials and lign it into acceptable tensor and lengths
# collate function to handle these ragged list
# ctc rquires the targets as [y₁ y₂ y₃ | y₁ y₂ | y₁ y₂ y₃ y₄] and uses target_lengths to split this back internally

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import h5py
import numpy as np
import torch

def collate_ctc(batch):
    lengths = torch.tensor([b["input_length"] for b in batch],dtype = torch.long)
    Tmax = int(lengths.max().item())
    D = int(batch[0]["inputs"].shape[1]) #inputs will be T,512 as abov

    #Now to add padding
    xp = torch.zeros((len(batch),Tmax,D),dtype=torch.float32)
    for i,b in enumerate(batch):
        t = b["inputs"].shape[0]
        xp[i,:t] = b["inputs"]

    out = {
        "inputs":xp,
        "input_lengths":lengths,
        "session": [b["session"] for b in batch],
        "block_num": torch.tensor([b["block_num"] for b in batch], dtype=torch.long),
        "trial_num": torch.tensor([b["trial_num"] for b in batch], dtype=torch.long),
        "file_path": [b["file_path"] for b in batch],
        "key": [b["key"] for b in batch],
    }

    if "targets" in batch[0]:
        target_lengths = torch.tensor([b["target_lengths"] for b in batch], dtype = torch.long)
        #need to also handle for cases where no label is there
        if int(target_lengths.sum().item()) > 0:
            targets = torch.cat([b["targets"] for b in batch], dim=0) #concatenated form as it is reqd for ctc(acr rows)
        else:
            targets = torch.zeros((0,), dtype=torch.long)

        out.update(
            {
                "targets": targets,
                "target_lengths": target_lengths,
                "transcript": [b.get("transcript", "") for b in batch],
            }
        )
        return out

# collate function for seq2seq or ce transducer style
# dont require concatenated targets rather in the form of (B,Umax), so we add padding to it

def collate_ce(batch,pad_val=0):
    out = collate_ctc(batch) #building from this only
    if "targets" not in out:
        return out
    
    target_lengths = out["target_lengths"]
    Umax = int(target_lengths.max().item()) if target_lengths.numel()>0 else 0

    yp = torch.full((len(batch),Umax), fill_value=pad_val,dtype=torch.long)
    ptr = 0
    for i, L in enumerate(target_lengths.tolist()):
        if L>0:
            yp[i,:L] = out["targets"][ptr:ptr+L]
            ptr+=L
    out["targets_padded"] = yp
    return yp

