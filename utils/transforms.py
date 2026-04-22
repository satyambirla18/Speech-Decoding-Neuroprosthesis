import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

#Transform functions that can be applied to our case

class Compose:
    def __init__(self,transforms):
        self.transforms = list(transforms)
    def __call__(self,x):
        for t in self.transforms:
            x = t(x)
        return x
    
#To clamp values to a range
class Clip:
    def __init__(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self,x):
        return np.clip(x,self.min_val,self.max_val)
    
#channels wise zscore 
class ZscoreNorm:
    def __init__(self,mean,std,eps=1e-6):
        self.mean = np.asarray(mean,dtype=np.float32).reshape(1,-1)
        self.std = np.asarray(std,dtype=np.float32).reshape(1,-1)
        self.eps = float(eps)
    def __call__(self,x):
        return (x - self.mean) / (self.std + self.eps)
    
#Znorming each sample by its own mean, std over time. Warning tho, may lose info accross sessions so need to be careful
class SampleZscoreNorm:
    def __init__(self,eps=1e-6):
        self.eps = float(eps)
    def __call__(self,x):
        mean = np.mean(x,axis=0,keepdims=True)
        std = np.std(x,axis=0,keepdims=True)
        return (x - mean) / (std + self.eps)
    
#applying log(1+x) elementwise, note requires features >=0 else fails
class Log1p:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(x, 0.0))
    
#Simple Moving average over time
class MovingAverageSmooth:
    def __init__(self, window: int = 3):
        if window < 1 or window % 2 == 0:
            raise ValueError("window must be odd and >= 1")
        self.window = int(window)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.window == 1:
            return x
        pad = self.window // 2
        xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        csum = np.cumsum(xp, axis=0)
        sm = (csum[self.window:] - csum[:-self.window]) / float(self.window)
        return sm.astype(np.float32)

#downsampling by stride

class TimeDownsample:
    def __init__(self, stride: int = 2):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = int(stride)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[:: self.stride]


class AddGaussianNoise:
    def __init__(self, sigma: float = 0.02, p: float = 0.5):
        self.sigma = float(sigma)
        self.p = float(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return x
        noise = np.random.randn(*x.shape).astype(np.float32) * self.sigma
        return x + noise

#Randomly zero out a fraction of channels.
class RandomChannelDropout:
    def __init__(self, drop_prob: float = 0.05, p: float = 0.5):
        self.drop_prob = float(drop_prob)
        self.p = float(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p or self.drop_prob <= 0:
            return x
        D = x.shape[1]
        mask = (np.random.rand(D) >= self.drop_prob).astype(np.float32)
        return x * mask[None, :]

#SpecAugment-style masking along time.
class RandomTimeMask:
    def __init__(self, max_width: int = 30, num_masks: int = 2, p: float = 0.5):
        self.max_width = int(max_width)
        self.num_masks = int(num_masks)
        self.p = float(p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return x
        T = x.shape[0]
        if T <= 1:
            return x
        for _ in range(self.num_masks):
            w = np.random.randint(1, min(self.max_width, T) + 1)
            t0 = np.random.randint(0, max(1, T - w))
            x[t0 : t0 + w] = 0.0
        return x


def compute_channel_mean_std(
    root_dir: str,
    max_samples: Optional[int] = None,
    max_time_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    ds = BtTDataset(
        root_dir=root_dir,
        split="train",
        target_kind="none",
        max_time_steps=max_time_steps,
        transform=None,
        return_transcript=False,
    )

    n_used = 0
    sum_x = None
    sum_x2 = None
    count = 0

    for i in range(len(ds)):
        x = ds[i]["inputs"].numpy()  
        if sum_x is None:
            sum_x = x.sum(axis=0)
            sum_x2 = (x * x).sum(axis=0)
        else:
            sum_x += x.sum(axis=0)
            sum_x2 += (x * x).sum(axis=0)
        count += x.shape[0]

        n_used += 1
        if max_samples is not None and n_used >= max_samples:
            break

    mean = sum_x / max(count, 1)
    var = (sum_x2 / max(count, 1)) - (mean * mean)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)

    ds.close()
    return mean.astype(np.float32), std.astype(np.float32)
