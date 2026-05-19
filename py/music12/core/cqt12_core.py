from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math

import numpy as np
import pandas as pd

try:
    import librosa
except Exception as e:
    librosa = None
    _LIBROSA_IMPORT_ERROR = e
else:
    _LIBROSA_IMPORT_ERROR = None

from music12.core.notation12 import (
    DEFAULT_ANCHOR_TOKEN,
    hz_to_token,
    normalize_token,
    token_to_abs_semitone_index,
)


@dataclass
class CQT12Params:
    a4_hz: float = 440.0
    anchor_token: str = DEFAULT_ANCHOR_TOKEN

    bins_per_octave: int = 144
    time_division_per_second: int = 60

    # lower analysis anchor in project coordinates
    fmin_token: str = "5.1"
    n_octaves: int = 7

    # selection per frame
    top_bins_per_frame: int = 12
    amplitude_floor_ratio: float = 0.03

    # librosa.cqt parameters
    filter_scale: float = 1.0
    sparsity: float = 0.01

    # candidate policy
    keep_all_selected_bins: bool = False


def _require_librosa() -> None:
    if librosa is None:
        raise ImportError(
            "librosa is required for cqt12_core.py but could not be imported. "
            f"Original import error: {_LIBROSA_IMPORT_ERROR}"
        )


def _token_to_hz(token: str, a4_hz: float, anchor_token: str) -> float:
    tok = normalize_token(token)
    anc = normalize_token(anchor_token)

    idx = token_to_abs_semitone_index(tok)
    anc_idx = token_to_abs_semitone_index(anc)

    semitone_delta = idx - anc_idx
    return float(a4_hz * (2.0 ** (semitone_delta / 12.0)))


def _build_cqt_frequencies(fmin_hz: float, n_bins: int, bins_per_octave: int) -> np.ndarray:
    k = np.arange(n_bins, dtype=float)
    return fmin_hz * (2.0 ** (k / float(bins_per_octave)))


def _freq_to_music12_cqt_position(
    freq_hz: float,
    params: CQT12Params,
) -> tuple[int, int]:
    """
    Return:
      octave_index_1based,
      step144_index_1based

    Here octave means coarse 144-bin group from fmin_token origin.
    """
    fmin_hz = _token_to_hz(params.fmin_token, params.a4_hz, params.anchor_token)
    if freq_hz <= 0 or fmin_hz <= 0:
        return 0, 0

    pos = params.bins_per_octave * math.log2(freq_hz / fmin_hz)

    k_round = int(round(pos))
    octave_index_1based = (k_round // params.bins_per_octave) + 1
    step144_index_1based = (k_round % params.bins_per_octave) + 1

    return octave_index_1based, step144_index_1based


def _freq_to_token12_coarse(freq_hz: float, params: CQT12Params) -> str:
    return normalize_token(
        hz_to_token(
            float(freq_hz),
            a4_hz=params.a4_hz,
            anchor_token=params.anchor_token,
            micro_depth=1,
            force_micro_dash_when_exact=True,
        )
    )


def _local_maxima_indices(x: np.ndarray) -> List[int]:
    idx: List[int] = []
    if x.size < 3:
        return idx

    for i in range(1, x.size - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            idx.append(i)
    return idx


def _pick_frame_bins(
    amp_col: np.ndarray,
    params: CQT12Params,
) -> List[int]:
    if amp_col.size == 0:
        return []

    frame_max = float(np.max(amp_col))
    if frame_max <= 0:
        return []

    threshold = frame_max * float(params.amplitude_floor_ratio)
    candidate_idx = np.where(amp_col >= threshold)[0].tolist()

    if not candidate_idx:
        return []

    if params.keep_all_selected_bins:
        candidate_idx = sorted(candidate_idx, key=lambda i: amp_col[i], reverse=True)
        return candidate_idx[: params.top_bins_per_frame]

    local_idx = set(_local_maxima_indices(amp_col))
    selected = [i for i in candidate_idx if i in local_idx]

    if not selected:
        selected = candidate_idx

    selected = sorted(selected, key=lambda i: amp_col[i], reverse=True)
    return selected[: params.top_bins_per_frame]


def build_cqt12_table(
    wav_path: str,
    params: Optional[CQT12Params] = None,
) -> pd.DataFrame:
    """
    Primary music12-oriented scanner.

    Output columns:
      - frame_id
      - t_sec
      - t60_index
      - rank
      - cqt_bin_index
      - octave_index_1based
      - step144_index_1based
      - freq_hz_est
      - amp
      - token12_coarse
    """
    _require_librosa()
    params = params or CQT12Params()

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if y is None or len(y) == 0:
        return pd.DataFrame(columns=[
            "frame_id",
            "t_sec",
            "t60_index",
            "rank",
            "cqt_bin_index",
            "octave_index_1based",
            "step144_index_1based",
            "freq_hz_est",
            "amp",
            "token12_coarse",
        ])

    hop_length = int(round(sr / float(params.time_division_per_second)))
    hop_length = max(hop_length, 1)

    fmin_hz = _token_to_hz(params.fmin_token, params.a4_hz, params.anchor_token)
    n_bins = int(params.n_octaves * params.bins_per_octave)

    C = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin_hz,
        n_bins=n_bins,
        bins_per_octave=params.bins_per_octave,
        filter_scale=params.filter_scale,
        sparsity=params.sparsity,
    )

    A = np.abs(C)
    freqs = _build_cqt_frequencies(
        fmin_hz=fmin_hz,
        n_bins=n_bins,
        bins_per_octave=params.bins_per_octave,
    )

    rows = []
    n_frames = A.shape[1]

    for frame_id in range(n_frames):
        amp_col = A[:, frame_id]
        picked = _pick_frame_bins(amp_col, params)

        t_sec = frame_id * hop_length / float(sr)
        t60_index = int(round(t_sec * params.time_division_per_second))

        for rank, bin_idx in enumerate(picked, start=1):
            freq_hz = float(freqs[bin_idx])
            amp = float(amp_col[bin_idx])

            octave_index_1based, step144_index_1based = _freq_to_music12_cqt_position(
                freq_hz=freq_hz,
                params=params,
            )

            rows.append({
                "frame_id": int(frame_id),
                "t_sec": float(t_sec),
                "t60_index": int(t60_index),
                "rank": int(rank),
                "cqt_bin_index": int(bin_idx),
                "octave_index_1based": int(octave_index_1based),
                "step144_index_1based": int(step144_index_1based),
                "freq_hz_est": float(freq_hz),
                "amp": float(amp),
                "token12_coarse": _freq_to_token12_coarse(freq_hz, params),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        by=["frame_id", "rank", "amp"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_cqt12_best_per_frame(
    wav_path: str,
    params: Optional[CQT12Params] = None,
) -> pd.DataFrame:
    df = build_cqt12_table(wav_path, params=params)
    if df.empty:
        return df

    return (
        df.sort_values(["frame_id", "amp"], ascending=[True, False])
          .groupby("frame_id", as_index=False)
          .head(1)
          .copy()
          .reset_index(drop=True)
    )