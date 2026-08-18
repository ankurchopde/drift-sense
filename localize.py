"""Drift-Sense ML Pattern Localizer (Tuned Random Forest Champion).

Stand-alone SEM spot localization inference pipeline loading model.joblib to rank
extracted SSD candidates via the verified 20-feature representation.

Inference Pipeline:
    Reference image (1000x1000, 1nm/px)
        +
    Search image (1000x1000, 10nm/px)
        ↓
    Classical FFT / SSD Correlation Map
        ↓
    Local Minima Candidate Extraction (Top-K, 5px Chebyshev separation)
        ↓
    Exact 20-Feature Engineering Matrix (X shape: [N, 20])
        ↓
    Tuned Random Forest Candidate Scoring (model.predict_proba[:, 1])
        ↓
    Top Candidate Selection (argmax RF probability)
        ↓
    Predicted Target Center (predicted_x, predicted_y)

Usage:
    python localize.py --reference <reference.png> --search <search.png>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Tuple

import joblib
import numpy as np
from PIL import Image

SEARCH_TO_REFERENCE_SCALE = 10
EXCLUSION_RADIUS_PX = 5
TOP_K_CANDIDATES = 100

FEATURE_NAMES = [
    "ssd_raw",                # 0  raw SSD score
    "center_x",               # 1  candidate center x
    "center_y",               # 2  candidate center y
    "rank",                   # 3  SSD-ascending rank (0=best)
    "difficulty_enc",          # 4  L0=0..L3=3 (default 3)
    "noise_enc",               # 5  CLEAN=0..EXTREME=4 (default 2)
    "noise_multiplier",        # 6  noise multiplier (default 1.0)
    "search_noise_std",        # 7  search noise std px (default 2.0)
    "ref_noise_std",           # 8  ref noise std px (default 0.5)
    "line_scan_corr",          # 9  line scan correlation (default 0.8)
    "ssd_ratio_to_best",       # 10 ssd / min(ssd) per sample (>=1)
    "ssd_gap_to_best",         # 11 ssd - min(ssd)
    "ssd_zscore",              # 12 z-score within sample
    "log_ssd",                 # 13 log(1 + ssd)
    "rank_pct",                # 14 rank / n_candidates
    "dist_to_center",          # 15 euclidean dist from (500, 500)
    "x_norm",                  # 16 center_x / 1000
    "y_norm",                  # 17 center_y / 1000
    "ssd_ratio_to_second",     # 18 ssd / ssd_of_rank1
    "ssd_ratio_to_median",     # 19 ssd / median(ssd) per sample
]


@dataclass(frozen=True)
class LocalizationOutput:
    predicted_x: float
    predicted_y: float
    confidence: float
    num_candidates: int
    runtime_ms: float
    model_name: str


def _load_grayscale(path: str | Path) -> Image.Image:
    """Load image converted to 8-bit grayscale."""
    with Image.open(path) as image:
        return image.convert("L")


def _downsample_reference(reference: Image.Image, scale: int = SEARCH_TO_REFERENCE_SCALE) -> Image.Image:
    """Downsample reference by factor 10 to match search pixel pitch (1nm -> 10nm)."""
    if reference.width % scale or reference.height % scale:
        raise ValueError("Reference dimensions must be divisible by sampling scale (10).")
    return reference.resize((reference.width // scale, reference.height // scale), Image.Resampling.BOX)


def _valid_window_sums(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return sums for every valid height-by-width window using integral images."""
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def _cross_correlation(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return valid 2D cross-correlation via deterministic FFT convolution."""
    height, width = search.shape
    template_height, template_width = template.shape
    fft_shape = (height + template_height - 1, width + template_width - 1)
    full = np.fft.irfft2(
        np.fft.rfft2(search, fft_shape) * np.fft.rfft2(template[::-1, ::-1], fft_shape),
        fft_shape,
    )
    return full[template_height - 1 : height, template_width - 1 : width]


def compute_ssd_map(reference_img: Image.Image, search_img: Image.Image) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Compute dense Sum-of-Squared-Differences (SSD) map across all valid positions."""
    ref_down = _downsample_reference(reference_img, SEARCH_TO_REFERENCE_SCALE)
    template = np.asarray(ref_down, dtype=np.float64)
    source = np.asarray(search_img, dtype=np.float64)
    th, tw = template.shape
    
    correlation = _cross_correlation(source, template)
    ssd = _valid_window_sums(source * source, th, tw) + np.sum(template * template) - 2.0 * correlation
    return np.maximum(ssd, 0.0), source, th, tw


def extract_candidate_peaks(ssd: np.ndarray, k: int = TOP_K_CANDIDATES, min_sep_px: int = EXCLUSION_RADIUS_PX) -> List[Tuple[int, int, float]]:
    """Extract Top-K spatially distinct local minima from the SSD map."""
    h, w = ssd.shape
    order = np.argsort(ssd, axis=None, kind="stable")
    
    # 8-neighbour local minimum mask
    pad = np.pad(ssd, 1, mode="edge")
    local_min = (
        (ssd <= pad[:-2, 1:-1])
        & (ssd <= pad[2:, 1:-1])
        & (ssd <= pad[1:-1, :-2])
        & (ssd <= pad[1:-1, 2:])
        & (ssd <= pad[:-2, :-2])
        & (ssd <= pad[:-2, 2:])
        & (ssd <= pad[2:, :-2])
        & (ssd <= pad[2:, 2:])
    )
    
    sep = max(1, int(min_sep_px))
    accepted: List[Tuple[int, int, float]] = []
    exclusion = np.zeros((h, w), dtype=bool)
    
    for flat_idx in order:
        y = int(flat_idx) // w
        x = int(flat_idx) % w
        if not local_min[y, x] or exclusion[y, x]:
            continue
            
        score = float(ssd[y, x])
        accepted.append((y, x, score))
        if len(accepted) >= k:
            break
            
        y0, y1 = max(0, y - sep), min(h, y + sep + 1)
        x0, x1 = max(0, x - sep), min(w, x + sep + 1)
        exclusion[y0:y1, x0:x1] = True
        
    return accepted


def extract_20_features(
    candidates: List[Tuple[int, int, float]],
    th: int,
    tw: int,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Construct the exact 20-feature matrix for Random Forest scoring."""
    n_cand = len(candidates)
    cands_sorted = sorted(candidates, key=lambda c: c[2])  # Sort by SSD ascending
    
    ssd_values = np.array([c[2] for c in cands_sorted], dtype=np.float64)
    ssd_min = ssd_values[0] if len(ssd_values) > 0 else 1.0
    ssd_second = ssd_values[1] if len(ssd_values) > 1 else ssd_min
    ssd_mean = ssd_values.mean() if len(ssd_values) > 0 else 1.0
    ssd_std = max(ssd_values.std(), 1e-10)
    ssd_median = np.median(ssd_values) if len(ssd_values) > 0 else 1.0
    
    ssd_safe = max(ssd_min, 1e-10)
    ssd_second_safe = max(ssd_second, 1e-10)
    ssd_median_safe = max(ssd_median, 1e-10)
    
    X_list = []
    centers = []
    
    for rank, (y, x, ssd) in enumerate(cands_sorted):
        cx = float(x + tw / 2.0)
        cy = float(y + th / 2.0)
        centers.append((cx, cy))
        
        # Exact 20-feature definition
        feat = np.array([
            ssd,                                         # 0  ssd_raw
            cx,                                          # 1  center_x
            cy,                                          # 2  center_y
            float(rank),                                 # 3  rank
            3.0,                                         # 4  difficulty_enc (L3)
            2.0,                                         # 5  noise_enc (MEDIUM)
            1.0,                                         # 6  noise_multiplier
            2.0,                                         # 7  search_noise_std
            0.5,                                         # 8  ref_noise_std
            0.8,                                         # 9  line_scan_corr
            ssd / ssd_safe,                              # 10 ssd_ratio_to_best
            ssd - ssd_min,                               # 11 ssd_gap_to_best
            (ssd - ssd_mean) / ssd_std,                  # 12 ssd_zscore
            np.log1p(ssd),                               # 13 log_ssd
            rank / max(n_cand, 1),                       # 14 rank_pct
            np.sqrt((cx - 500.0)**2 + (cy - 500.0)**2),  # 15 dist_to_center
            cx / 1000.0,                                 # 16 x_norm
            cy / 1000.0,                                 # 17 y_norm
            ssd / ssd_second_safe,                       # 18 ssd_ratio_to_second
            ssd / ssd_median_safe,                       # 19 ssd_ratio_to_median
        ], dtype=np.float64)
        X_list.append(feat)
        
    X = np.array(X_list, dtype=np.float64)
    return X, centers


def localize(reference_path: str | Path, search_path: str | Path, model_path: str | Path | None = None) -> LocalizationOutput:
    """Run full ML spot localization on reference and search images."""
    t0 = time.perf_counter()
    
    ref_p = Path(reference_path)
    srch_p = Path(search_path)
    if not ref_p.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_p}")
    if not srch_p.exists():
        raise FileNotFoundError(f"Search image not found: {srch_p}")
        
    # 1. Resolve model path
    if model_path is None:
        candidates = ["model.joblib", "tuned_rf_model.joblib"]
        model_path = None
        for name in candidates:
            cand_p = Path(__file__).resolve().parent / name
            if cand_p.exists():
                model_path = cand_p
                break
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "model.joblib"
    else:
        model_path = Path(model_path)
        
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
        
    # 2. Load trained Tuned RF model
    model = joblib.load(model_path)
    
    # 3. Load images
    ref_img = _load_grayscale(ref_p)
    srch_img = _load_grayscale(srch_p)
    
    # 4. Generate SSD correlation map
    ssd, _, th, tw = compute_ssd_map(ref_img, srch_img)
    
    # 5. Extract distinct candidate local minima
    candidates = extract_candidate_peaks(ssd, k=TOP_K_CANDIDATES, min_sep_px=EXCLUSION_RADIUS_PX)
    if not candidates:
        raise RuntimeError("No candidate peaks could be extracted from SSD map.")
        
    # 6. Extract exact 20-feature matrix
    X, centers = extract_20_features(candidates, th, tw)
    assert X.shape[1] == getattr(model, "n_features_in_", 20), (
        f"Feature count mismatch: got {X.shape[1]}, expected {model.n_features_in_}"
    )
    
    # 7. Score all candidates using RF positive class probability
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X).astype(float)
        
    # 8. Rank candidates and pick top-1 prediction
    best_idx = int(np.argmax(scores))
    pred_x, pred_y = centers[best_idx]
    confidence = float(scores[best_idx])
    
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    
    return LocalizationOutput(
        predicted_x=round(pred_x, 2),
        predicted_y=round(pred_y, 2),
        confidence=round(confidence, 4),
        num_candidates=len(candidates),
        runtime_ms=round(runtime_ms, 2),
        model_name=model.__class__.__name__,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drift-Sense Final ML Spot Localizer (Tuned Random Forest Champion)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        required=True,
        help="Path to high-magnification Reference SEM image (1000x1000 PNG)",
    )
    parser.add_argument(
        "--search",
        "-s",
        type=str,
        required=True,
        help="Path to overview Search SEM image (1000x1000 PNG)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to model.joblib checkpoint (defaults to model.joblib in script directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format only",
    )
    
    args = parser.parse_args()
    
    result = localize(args.reference, args.search, args.model)
    
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print(f"\n{'='*65}")
        print(f"  DRIFT-SENSE ML LOCALIZATION RESULT")
        print(f"{'='*65}")
        print(f"  Predicted Center  : ({result.predicted_x}, {result.predicted_y})")
        print(f"  RF Confidence     : {result.confidence:.4f}")
        print(f"  Candidates Scored : {result.num_candidates}")
        print(f"  Model             : {result.model_name}")
        print(f"  Runtime           : {result.runtime_ms:.1f} ms")
        print(f"{'='*65}\n")
        print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
