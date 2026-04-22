from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional
import numpy as np

LOGIT_TO_PHONEME: List[str] = [
    "BLANK",
    "AA","AE","AH","AO","AW",
    "AY","B","CH","D","DH",
    "EH","ER","EY","F","G",
    "HH","IH","IY","JH","K",
    "L","M","N","NG","OW",
    "OY","P","R","S","SH",
    "T","TH","UH","UW","V",
    "W","Y","Z","ZH",
    " | ",
]

PHONEME_TO_LOGIT = {p:i for i,p in enumerate(LOGIT_TO_PHONEME)}

def safe_int(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (bytes, bytearray)):
        try:
            return int(x.decode("utf-8"))
        except Exception:
            return int.from_bytes(x, "little", signed=False)
    return int(x)

def decode_ascii_array(arr) -> str:
    if arr is None:
        return ""
    if isinstance(arr, (bytes, bytearray)):
        try:
            return arr.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    a = np.asarray(arr)
    if a.dtype.kind in {"S", "U"}:
        try:
            return str(a.tolist())
        except Exception:
            return ""
    chars = []
    for v in a.reshape(-1):
        iv = safe_int(v)
        if iv == 0:
            continue
        if 0 <= iv <= 0x10FFFF:
            try:
                chars.append(chr(iv))
            except Exception:
                pass
    return "".join(chars).strip()

def ctc_greedy_decode(ids: Sequence[int], blank_id: int = 0) -> List[int]:
    out: List[int] = []
    prev = None
    for i in ids:
        if i == prev:
            continue
        if i != blank_id:
            out.append(int(i))
        prev = int(i)
    return out

def ids_to_phonemes(ids: Sequence[int]) -> List[str]:
    return [LOGIT_TO_PHONEME[int(i)] if 0 <= int(i) < len(LOGIT_TO_PHONEME) else "<?>"
            for i in ids]

def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost
            )
            prev = cur
    return dp[m]
