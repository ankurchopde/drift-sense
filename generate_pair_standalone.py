"""Drift-Sense reference/search pair generator -- standalone single-file build.

Consolidates config.py + dram_geometry.py + rasterize.py + imaging.py +
noise.py + sem_intensity.py + generate_pair.py into one file with no
package-relative imports, so it can be copied and run anywhere with just
numpy / scipy / Pillow installed.

Generation math is byte-for-byte unchanged from the original modules for the
same (seed, difficulty/knob) inputs -- only the CLI/output surface differs:
instead of a fixed output_dir containing reference.png/search.png/metadata.json,
you pass the reference/search (and optionally metadata) file paths directly,
matching the --reference/--search style used by localize.py.

Usage:
    python generate_pair_standalone.py --reference out/reference.png --search out/search.png
    python generate_pair_standalone.py --reference r.png --search s.png --seed 12345 --difficulty L3
    python generate_pair_standalone.py --reference r.png --search s.png --metadata out/meta.json --noise
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

# =============================================================================
# config.py -- constants for the B2.1 clean, physical DRAM baseline
# =============================================================================

MASTER_SIZE_NM = 10_000
HALF_PITCH_NM = 20
CELL_FACTOR = "6F"

CELL_WIDTH_NM = 2 * HALF_PITCH_NM
CELL_HEIGHT_NM = 3 * HALF_PITCH_NM

DEFAULT_SEED = 2101
GENERATOR_CONFIG_VERSION = "B8.1"

DEFAULT_VARIATION_AMPLITUDE_NM = 0.0
DEFAULT_CORRELATION_LENGTH_NM = 20.0

DEFAULT_PSF_SIGMA_NM = 5.0

DEFAULT_CDU_DW_WL_NM = 0.0
DEFAULT_CDU_DW_BL_NM = 0.0
DEFAULT_CDU_DS_BLC_NM = 0.0
DEFAULT_CDU_DS_SNC_NM = 0.0

DEFAULT_OVERLAY_DX_SNC_NM = 0.0
DEFAULT_OVERLAY_DY_SNC_NM = 0.0


@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    description: str
    amplitude_nm: float = 0.0
    psf_sigma_nm: float = DEFAULT_PSF_SIGMA_NM
    noise_enabled: bool = False
    line_scan_correlation_strength: float = 0.0
    cdu_dw_wl_nm: float = 0.0
    cdu_dw_bl_nm: float = 0.0
    cdu_ds_blc_nm: float = 0.0
    cdu_ds_snc_nm: float = 0.0
    overlay_dx_snc_nm: float = 0.0
    overlay_dy_snc_nm: float = 0.0


DIFFICULTY_LEVELS = {
    "L0": DifficultyConfig(
        name="L0",
        description="CLEAN: geometry only, exact 1/10 nm-per-pixel relation, no process variation, no noise",
        amplitude_nm=0.0,
        psf_sigma_nm=0.0,
        noise_enabled=False,
        line_scan_correlation_strength=0.0,
    ),
    "L1": DifficultyConfig(
        name="L1",
        description="IMAGING: existing PSF + existing intensity mapping + existing required SEM noise",
        amplitude_nm=0.0,
        psf_sigma_nm=DEFAULT_PSF_SIGMA_NM,
        noise_enabled=True,
        line_scan_correlation_strength=0.8,
    ),
    "L2": DifficultyConfig(
        name="L2",
        description="PROCESS: L1 + B3.1 CDU + B3.2 overlay",
        amplitude_nm=0.0,
        psf_sigma_nm=DEFAULT_PSF_SIGMA_NM,
        noise_enabled=True,
        line_scan_correlation_strength=0.8,
        cdu_dw_wl_nm=2.0,
        cdu_dw_bl_nm=-1.0,
        cdu_ds_blc_nm=1.0,
        cdu_ds_snc_nm=1.0,
        overlay_dx_snc_nm=3.0,
        overlay_dy_snc_nm=-2.0,
    ),
    "L3": DifficultyConfig(
        name="L3",
        description="FULL CURRENT MODEL: all promoted effects (PSF, noise, B2.5 LER, CDU, overlay, B2.7-R)",
        amplitude_nm=0.5,
        psf_sigma_nm=DEFAULT_PSF_SIGMA_NM,
        noise_enabled=True,
        line_scan_correlation_strength=0.8,
        cdu_dw_wl_nm=2.0,
        cdu_dw_bl_nm=-1.0,
        cdu_ds_blc_nm=1.0,
        cdu_ds_snc_nm=1.0,
        overlay_dx_snc_nm=3.0,
        overlay_dy_snc_nm=-2.0,
    ),
}

LAYER_ORDER = ("wordline", "bitline", "bitline_contact", "storage_node_contact")

REFERENCE_MAGNIFICATION_X = 100
SEARCH_MAGNIFICATION_X = 10


# =============================================================================
# shared helper (identical copy used across the original modules)
# =============================================================================

def _deterministic_seed(*components: object) -> int:
    """Generate a stable 64-bit integer seed from arbitrary identifier components."""
    payload = "|".join(str(c) for c in components).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# =============================================================================
# dram_geometry.py -- compact parametric geometry for the B2.1 clean master
# =============================================================================

def _repeat_count(size_nm: int, period_nm: int) -> int:
    return (size_nm + period_nm - 1) // period_nm


def _array_regions() -> list[dict[str, Any]]:
    """Return nine finite, separated DRAM array blocks in physical nm."""
    block_width_nm = 2_800
    block_height_nm = 2_640
    x_origins = (300, 3_650, 7_000)
    y_origins = (300, 3_680, 7_060)
    regions = []
    for row, y in enumerate(y_origins):
        for column, x in enumerate(x_origins):
            regions.append(
                {
                    "id": f"array_{row}_{column}",
                    "bounds_nm": [x, y, x + block_width_nm, y + block_height_nm],
                    "cell_grid": {"columns": block_width_nm // CELL_WIDTH_NM, "rows": block_height_nm // CELL_HEIGHT_NM},
                }
            )
    return regions


def _regional_primitives(regions: list[dict[str, Any]], layer: str) -> list[dict[str, Any]]:
    """Build repeated primitive definitions for one layer in finite arrays."""
    primitives = []
    for region in regions:
        left, top, right, bottom = region["bounds_nm"]
        if layer == "wordline":
            rectangle, period = [left, top, right, top + HALF_PITCH_NM], [0, CELL_HEIGHT_NM]
        elif layer == "bitline":
            rectangle, period = [left, top, left + HALF_PITCH_NM, bottom], [CELL_WIDTH_NM, 0]
        elif layer == "bitline_contact":
            rectangle, period = [left + 4, top + 24, left + 16, top + 36], [CELL_WIDTH_NM, CELL_HEIGHT_NM]
        elif layer == "storage_node_contact":
            rectangle, period = [left + 24, top + 22, left + 38, top + 38], [CELL_WIDTH_NM, CELL_HEIGHT_NM]
        else:
            raise ValueError(f"unknown regional layer: {layer}")
        primitives.append({"bounds_nm": [left, top, right, bottom], "rectangle_nm": rectangle, "period_nm": period})
    return primitives


def layout_sha256(master: dict[str, Any]) -> str:
    """Hash the canonical geometry, excluding its derived hash field."""
    canonical = {key: value for key, value in master.items() if key != "layout_sha256"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_clean_master(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Return the deterministic, compact hierarchical B2.1 master."""
    regions = _array_regions()
    total_cells = sum(region["cell_grid"]["columns"] * region["cell_grid"]["rows"] for region in regions)
    routing_rectangles = [
        [3_250, 0, 3_390, MASTER_SIZE_NM], [6_650, 0, 6_790, MASTER_SIZE_NM],
        [0, 3_200, MASTER_SIZE_NM, 3_340], [0, 6_580, MASTER_SIZE_NM, 6_720],
        [0, 120, MASTER_SIZE_NM, 180], [0, 9_820, MASTER_SIZE_NM, 9_880],
    ]
    for region in regions:
        left, top, right, bottom = region["bounds_nm"]
        routing_rectangles.extend(
            [[left + offset, top, left + offset + 60, bottom] for offset in (600, 1_400, 2_200)]
        )
        routing_rectangles.extend(
            [[left, top + offset, right, top + offset + 60] for offset in (600, 1_320, 1_980)]
        )

    layers = {
        "wordline": {
            "kind": "regional_periodic_rectangles",
            "orientation": "horizontal",
            "primitives": _regional_primitives(regions, "wordline"),
            "count": sum(region["cell_grid"]["rows"] for region in regions),
        },
        "bitline": {
            "kind": "regional_periodic_rectangles",
            "orientation": "vertical",
            "primitives": _regional_primitives(regions, "bitline"),
            "count": sum(region["cell_grid"]["columns"] for region in regions),
        },
        "bitline_contact": {
            "kind": "regional_periodic_rectangles",
            "orientation": "contact",
            "primitives": _regional_primitives(regions, "bitline_contact"),
            "count": total_cells,
        },
        "storage_node_contact": {
            "kind": "regional_periodic_rectangles",
            "orientation": "contact",
            "primitives": _regional_primitives(regions, "storage_node_contact"),
            "count": total_cells,
        },
        "routing": {
            "kind": "static_rectangles",
            "orientation": "routing",
            "rectangles_nm": routing_rectangles,
            "count": len(routing_rectangles),
        },
    }

    master = {
        "seed": seed,
        "units": "nm",
        "master_size_nm": [MASTER_SIZE_NM, MASTER_SIZE_NM],
        "half_pitch_nm": HALF_PITCH_NM,
        "cell_factor": CELL_FACTOR,
        "implementation_factor": [2, 3],
        "cell_width_nm": CELL_WIDTH_NM,
        "cell_height_nm": CELL_HEIGHT_NM,
        "geometry": "manhattan_rectilinear",
        "representation": "parametric_hierarchical_rectangles",
        "array_regions": regions,
        "array_region_count": len(regions),
        "layer_order": ["routing", *LAYER_ORDER],
        "layers": layers,
    }
    master["layout_sha256"] = layout_sha256(master)
    return master


# =============================================================================
# rasterize.py -- only the piece generate_sample actually needs
# =============================================================================

# Lower values represent more material coverage in this idealized clean image.
INTENSITY_MAPPING = {
    "background": 245,
    "wordline": 180,
    "bitline": 145,
    "bitline_contact": 85,
    "storage_node_contact": 55,
    "routing": 105,
}


# =============================================================================
# imaging.py -- correlated geometry variation + Gaussian PSF forward model
# =============================================================================

CDU_MODEL = "symmetric_width_bias_per_level"
OVERLAY_MODEL = "layer_level_relative_misregistration"


def cdu_symmetric_rect(
    x0: float, y0: float, x1: float, y1: float, *, delta_width: float, delta_height: float,
) -> tuple[float, float, float, float]:
    """B3.1 CDU: return a rectangle resized symmetrically about its center."""
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half_w = (x1 - x0 + delta_width) / 2.0
    half_h = (y1 - y0 + delta_height) / 2.0
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def build_cdu_metadata(dw_wl_nm: float, dw_bl_nm: float, dS_blc_nm: float, dS_snc_nm: float) -> dict:
    return {
        "enabled": True,
        "model": CDU_MODEL,
        "applied_to": "layer_geometry_dimensions_before_ler_edge_profiles_in_render_high_res_master",
        "layer_centers_preserved": True,
        "cell_pitch_preserved": True,
        "distinct_from_ler": True,
        "ler_channel": "independent_per_edge_displacement",
        "cdu_channel": "symmetric_width_size_bias",
        "dw_wl_nm": float(dw_wl_nm),
        "dw_bl_nm": float(dw_bl_nm),
        "dS_blc_nm": float(dS_blc_nm),
        "dS_snc_nm": float(dS_snc_nm),
        "cdu_parameter_provenance": "engineering-selected",
        "cdu_physical_basis": "literature-supported",
        "literature_basis": (
            "ITRS (2015) More Moore / IRDS (2017, 2024) More Moore identify Critical Dimension "
            "Uniformity (CDU) as a key nanoscale manufacturing and metrology challenge. Bunday et al. "
            "(L01, L02) and Constantoudis et al. (L03) give direct CD-SEM LER/CD metrology evidence. "
            "The exact numerical deltas are ENGINEERING-SELECTED experiment values, NOT literature "
            "values; they are not claimed to reproduce any specific wafer-fab CDU budget."
        ),
    }


def build_overlay_metadata(dx_snc_nm: float, dy_snc_nm: float) -> dict:
    return {
        "enabled": True,
        "model": OVERLAY_MODEL,
        "applied_to": "storage_node_contact_layer_rendering",
        "isolated_from": ["cdu", "ler", "psf", "noise", "line_scan_correlation", "target_gt"],
        "common_mode_reference_and_search": True,
        "gt_invariant": True,
        "cell_pitch_preserved": True,
        "geometry_dimensions_preserved": True,
        "design_centers_preserved": True,
        "printed_snc_center_delta_nm": [float(dx_snc_nm), float(dy_snc_nm)],
        "overlay_dx_snc_nm": float(dx_snc_nm),
        "overlay_dy_snc_nm": float(dy_snc_nm),
        "overlay_parameter_provenance": "engineering-selected",
        "overlay_physical_basis": "literature-supported",
        "literature_basis": (
            "Fukagawa et al. (O01), Sullivan & Shin (O02), and Graff et al. (2023, O03) support "
            "the physical existence of lithographic overlay errors and their systematic/random "
            "decomposition at few-nm scale. This B3.2 model represents the storage-node-contact "
            "level printed with a deterministic relative misregistration (dx, dy) with respect to "
            "the wordline/bitline/bitline-contact grid. The exact numerical offsets are "
            "ENGINEERING-SELECTED experiment values, NOT literature values."
        ),
    }


def generate_correlated_profile(
    length: int, amplitude_nm: float, correlation_length_nm: float, *identity: object,
) -> np.ndarray:
    """Generate a 1D Gaussian-correlated continuous edge displacement profile."""
    if length <= 0 or amplitude_nm <= 0.0:
        return np.zeros(length, dtype=np.float64)

    seed_int = _deterministic_seed(*identity)
    rng = np.random.default_rng(seed_int)

    pad = int(max(3 * correlation_length_nm, 10))
    raw_noise = rng.standard_normal(length + 2 * pad)

    smooth = ndi.gaussian_filter1d(raw_noise, sigma=correlation_length_nm, mode="reflect")
    cropped = smooth[pad: pad + length]

    std = float(np.std(cropped))
    if std > 1e-9:
        return cropped * (amplitude_nm / std)
    return np.zeros(length, dtype=np.float64)


def apply_gaussian_psf(image: np.ndarray, sigma_nm: float, nm_per_pixel: float = 1.0) -> np.ndarray:
    """Apply a deterministic 2D Gaussian PSF convolution in physical nanometres."""
    if sigma_nm <= 0.0:
        return image.astype(np.float64, copy=True)
    sigma_px = sigma_nm / nm_per_pixel
    return ndi.gaussian_filter(image.astype(np.float64), sigma=sigma_px, mode="nearest")


def area_downsample(image: np.ndarray, factor: int = 10) -> np.ndarray:
    """Downsample a 2D image array by exact 2D area-averaging over factor-by-factor blocks."""
    if factor <= 0:
        raise ValueError("downsample factor must be positive")
    if factor == 1:
        return image.astype(np.float64, copy=True)
    height, width = image.shape[:2]
    if height % factor != 0 or width % factor != 0:
        raise ValueError("image dimensions must be exactly divisible by factor")
    reshaped = image.reshape(height // factor, factor, width // factor, factor)
    return reshaped.mean(axis=(1, 3))


def render_high_res_master(
    master: dict[str, Any],
    amplitude_nm: float = 0.0,
    correlation_length_nm: float = 20.0,
    seed: int = 2026,
    *,
    cdu_dw_wl_nm: float = 0.0,
    cdu_dw_bl_nm: float = 0.0,
    cdu_ds_blc_nm: float = 0.0,
    cdu_ds_snc_nm: float = 0.0,
    overlay_dx_snc_nm: float = 0.0,
    overlay_dy_snc_nm: float = 0.0,
) -> np.ndarray:
    """Render the full 10,000 x 10,000 master at 1 nm/px with optional LER/LWR/CDU/overlay variation."""
    w, h = master["master_size_nm"]
    canvas = np.full((h, w), INTENSITY_MAPPING["background"], dtype=np.uint8)

    for rect in master["layers"]["routing"]["rectangles_nm"]:
        x0, y0, x1, y1 = rect
        canvas[y0:y1, x0:x1] = INTENSITY_MAPPING["routing"]

    for prim_idx, prim in enumerate(master["layers"]["bitline"]["primitives"]):
        bx0, by0, bx1, by1 = prim["bounds_nm"]
        rx0, ry0, rx1, ry1 = prim["rectangle_nm"]
        period_x = prim["period_nm"][0]
        line_length = by1 - by0
        num_lines = (bx1 - bx0) // period_x
        nx0, ny0, nx1, ny1 = cdu_symmetric_rect(
            rx0, ry0, rx1, ry1, delta_width=cdu_dw_bl_nm, delta_height=0.0
        )
        for ix in range(num_lines):
            x_nom = nx0 + ix * period_x
            x_nom_right = nx1 + ix * period_x
            dl = generate_correlated_profile(
                line_length, amplitude_nm, correlation_length_nm, seed, "bitline", prim_idx, ix, "left"
            )
            dr = generate_correlated_profile(
                line_length, amplitude_nm, correlation_length_nm, seed, "bitline", prim_idx, ix, "right"
            )
            x_left_arr = np.clip(np.round(x_nom + dl).astype(int), 0, w - 1)
            x_right_arr = np.clip(np.round(x_nom_right + dr).astype(int), 0, w)
            for yi, y in enumerate(range(by0, by1)):
                xl = x_left_arr[yi]
                xr = max(xl + 1, x_right_arr[yi])
                canvas[y, xl:xr] = INTENSITY_MAPPING["bitline"]

    for prim_idx, prim in enumerate(master["layers"]["wordline"]["primitives"]):
        bx0, by0, bx1, by1 = prim["bounds_nm"]
        rx0, ry0, rx1, ry1 = prim["rectangle_nm"]
        period_y = prim["period_nm"][1]
        line_length = rx1 - rx0
        num_lines = (by1 - by0) // period_y
        nx0, ny0, nx1, ny1 = cdu_symmetric_rect(
            rx0, ry0, rx1, ry1, delta_width=0.0, delta_height=cdu_dw_wl_nm
        )
        for iy in range(num_lines):
            y_nom = ny0 + iy * period_y
            y_nom_bot = ny1 + iy * period_y
            dt = generate_correlated_profile(
                line_length, amplitude_nm, correlation_length_nm, seed, "wordline", prim_idx, iy, "top"
            )
            db = generate_correlated_profile(
                line_length, amplitude_nm, correlation_length_nm, seed, "wordline", prim_idx, iy, "bot"
            )
            y_top_arr = np.clip(np.round(y_nom + dt).astype(int), 0, h - 1)
            y_bot_arr = np.clip(np.round(y_nom_bot + db).astype(int), 0, h)
            for xi, x in enumerate(range(rx0, rx1)):
                yt = y_top_arr[xi]
                yb = max(yt + 1, y_bot_arr[xi])
                canvas[yt:yb, x] = INTENSITY_MAPPING["wordline"]

    for layer_name in ["bitline_contact", "storage_node_contact"]:
        val = INTENSITY_MAPPING[layer_name]
        dS = cdu_ds_blc_nm if layer_name == "bitline_contact" else cdu_ds_snc_nm
        for prim_idx, prim in enumerate(master["layers"][layer_name]["primitives"]):
            bx0, by0, bx1, by1 = prim["bounds_nm"]
            rx0, ry0, rx1, ry1 = prim["rectangle_nm"]
            px, py = prim["period_nm"]
            nx0, ny0, nx1, ny1 = cdu_symmetric_rect(
                rx0, ry0, rx1, ry1, delta_width=dS, delta_height=dS
            )
            if layer_name == "storage_node_contact" and (
                overlay_dx_snc_nm != 0.0 or overlay_dy_snc_nm != 0.0
            ):
                nx0 += overlay_dx_snc_nm
                ny0 += overlay_dy_snc_nm
                nx1 += overlay_dx_snc_nm
                ny1 += overlay_dy_snc_nm
            cw, ch = nx1 - nx0, ny1 - ny0
            nx = (bx1 - bx0) // px
            ny = (by1 - by0) // py
            for ix in range(nx):
                for iy in range(ny):
                    cx0 = nx0 + ix * px
                    cy0 = ny0 + iy * py
                    if amplitude_nm > 0.0:
                        s = _deterministic_seed(seed, layer_name, prim_idx, ix, iy)
                        rng = np.random.default_rng(s)
                        shifts = rng.normal(0, amplitude_nm, size=4)
                        cx0_v = int(np.clip(np.round(cx0 + shifts[0]), 0, w - 1))
                        cy0_v = int(np.clip(np.round(cy0 + shifts[1]), 0, h - 1))
                        cx1_v = int(np.clip(np.round(cx0 + cw + shifts[2]), cx0_v + 1, w))
                        cy1_v = int(np.clip(np.round(cy0 + ch + shifts[3]), cy0_v + 1, h))
                    else:
                        cx0_v = int(np.clip(np.round(cx0), 0, w - 1))
                        cy0_v = int(np.clip(np.round(cy0), 0, h - 1))
                        cx1_v = int(np.clip(np.round(cx0 + cw), cx0_v + 1, w))
                        cy1_v = int(np.clip(np.round(cy0 + ch), cy0_v + 1, h))
                    canvas[cy0_v:cy1_v, cx0_v:cx1_v] = val

    return canvas


# =============================================================================
# noise.py -- B2.6a/B2.6b Poisson-Gaussian SEM noise + line-scan correlation
# =============================================================================

REFERENCE_STREAM_ID = "reference"
SEARCH_STREAM_ID = "search"

DEFAULT_REFERENCE_GAIN_E_PER_INTENSITY = 25.0
DEFAULT_REFERENCE_DETECTOR_SIGMA_INTENSITY = 0.5
DEFAULT_SEARCH_GAIN_E_PER_INTENSITY = 1.0
DEFAULT_SEARCH_DETECTOR_SIGMA_INTENSITY = 2.0

DEFAULT_LINE_SCAN_CORRELATION_STRENGTH = 0.8
DEFAULT_REFERENCE_LINE_SCAN_CORRELATION_LENGTH_PX = 20.0
DEFAULT_SEARCH_LINE_SCAN_CORRELATION_LENGTH_PX = 2.0
DEFAULT_SCAN_AXIS = 1

NOISE_MODEL = "poisson_gaussian"
LINE_SCAN_CORRELATION_MODEL = "1d_gaussian_along_scan_axis"


@dataclass(frozen=True)
class NoiseParameters:
    enabled: bool
    model: str
    stream_id: str
    gain_e_per_intensity: float
    detector_sigma_intensity: float

    def seed(self, master_seed: int) -> int:
        return noise_stream_seed(master_seed, self.stream_id)

    def to_dict(self, master_seed: int) -> dict:
        return {
            "enabled": bool(self.enabled),
            "model": self.model,
            "stream_id": self.stream_id,
            "seed_derivation": "sha256(join('|', master_seed, 'noise', stream_id))[:8] bytes as big-endian int",
            "seed": self.seed(master_seed),
            "gain_e_per_intensity": float(self.gain_e_per_intensity),
            "detector_sigma_intensity": float(self.detector_sigma_intensity),
        }


DEFAULT_REFERENCE_NOISE = NoiseParameters(
    enabled=True, model=NOISE_MODEL, stream_id=REFERENCE_STREAM_ID,
    gain_e_per_intensity=DEFAULT_REFERENCE_GAIN_E_PER_INTENSITY,
    detector_sigma_intensity=DEFAULT_REFERENCE_DETECTOR_SIGMA_INTENSITY,
)

DEFAULT_SEARCH_NOISE = NoiseParameters(
    enabled=True, model=NOISE_MODEL, stream_id=SEARCH_STREAM_ID,
    gain_e_per_intensity=DEFAULT_SEARCH_GAIN_E_PER_INTENSITY,
    detector_sigma_intensity=DEFAULT_SEARCH_DETECTOR_SIGMA_INTENSITY,
)


def noise_stream_seed(master_seed: int, stream_id: str) -> int:
    return _deterministic_seed(master_seed, "noise", stream_id)


def apply_poisson_gaussian_noise(
    image: np.ndarray, *, seed: int, gain_e_per_intensity: float, detector_sigma_intensity: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lam = np.clip(image, 0.0, None) * gain_e_per_intensity
    counts = rng.poisson(lam)
    noisy = counts / gain_e_per_intensity
    noisy = noisy + rng.normal(0.0, detector_sigma_intensity, size=noisy.shape)
    return noisy


def apply_line_scan_correlation(
    noise_field: np.ndarray, *, strength: float, correlation_length_px: float, scan_axis: int = DEFAULT_SCAN_AXIS,
) -> np.ndarray:
    if strength <= 0.0 or correlation_length_px <= 0.0:
        return noise_field
    smooth = ndi.gaussian_filter1d(noise_field, sigma=correlation_length_px, axis=scan_axis, mode="nearest")
    s_white = float(np.std(noise_field))
    s_corr = float(np.std(smooth))
    if s_white > 1e-12 and s_corr > 1e-12:
        smooth = smooth * (s_white / s_corr)
    blended = (1.0 - strength) * noise_field + strength * smooth
    s_blend = float(np.std(blended))
    if s_white > 1e-12 and s_blend > 1e-12:
        blended = blended * (s_white / s_blend)
    return blended


def apply_sem_noise(
    image: np.ndarray, *, seed: int, gain_e_per_intensity: float, detector_sigma_intensity: float,
    line_scan_correlation_strength: float = 0.0, line_scan_correlation_length_px: float = 1.0,
    scan_axis: int = DEFAULT_SCAN_AXIS,
) -> np.ndarray:
    noisy = apply_poisson_gaussian_noise(
        image, seed=seed, gain_e_per_intensity=gain_e_per_intensity,
        detector_sigma_intensity=detector_sigma_intensity,
    )
    if line_scan_correlation_strength > 0.0:
        residual = noisy - image
        residual = apply_line_scan_correlation(
            residual, strength=line_scan_correlation_strength,
            correlation_length_px=line_scan_correlation_length_px, scan_axis=scan_axis,
        )
        noisy = image + residual
    return noisy


def build_line_scan_correlation_metadata(
    strength: float, reference_length_px: float, search_length_px: float, scan_axis: int,
) -> dict | str:
    if strength <= 0.0:
        return "not_applied_b26a"
    return {
        "enabled": True,
        "model": LINE_SCAN_CORRELATION_MODEL,
        "applied_to": "stochastic_noise_realization_only_not_deterministic_image",
        "strength": float(strength),
        "reference_length_px": float(reference_length_px),
        "search_length_px": float(search_length_px),
        "scan_axis": int(scan_axis),
        "scan_axis_meaning": "SEM horizontal raster scan direction (along image rows); 1D correlation only",
        "renormalization": "smoothed field rescaled to original std so the configured B2.6a noise level is preserved",
        "literature_basis": (
            "Roels et al. (2018) justify the existence and modeling of noise correlation "
            "associated with SEM line scanning (operator C in y = Hx + D(x) C n). The paper does "
            "NOT specify our correlation strength, kernel, or correlation length; strength and "
            "lengths here are engineering/test parameters."
        ),
    }


def build_noise_metadata(
    master_seed: int, reference: NoiseParameters, search: NoiseParameters,
    line_scan_correlation_strength: float = 0.0,
    reference_line_scan_correlation_length_px: float = DEFAULT_REFERENCE_LINE_SCAN_CORRELATION_LENGTH_PX,
    search_line_scan_correlation_length_px: float = DEFAULT_SEARCH_LINE_SCAN_CORRELATION_LENGTH_PX,
    scan_axis: int = DEFAULT_SCAN_AXIS,
) -> dict:
    return {
        "enabled": bool(reference.enabled or search.enabled),
        "model": NOISE_MODEL,
        "applied_after": "deterministic_image_formation_geometry_psf_downsampling",
        "independent_realizations": True,
        "independent_stream_ids": [reference.stream_id, search.stream_id],
        "independent_seeds": [reference.seed(master_seed), search.seed(master_seed)],
        "line_scan_correlation": build_line_scan_correlation_metadata(
            line_scan_correlation_strength, reference_line_scan_correlation_length_px,
            search_line_scan_correlation_length_px, scan_axis,
        ),
        "literature_basis": (
            "Roels et al. (2018), 'Bayesian Deconvolution of Scanning Electron Microscopy "
            "Images Using Point-spread Function Estimation and Non-local Regularization', IEEE "
            "TIP: models SEM observations as a mixed Poisson-Gaussian process with "
            "signal-dependent (shot) and signal-independent (detector) components, and discusses "
            "correlated noise from SEM line scanning. We adopt that conceptual model "
            "y = Hx + D(x) C n. The paper does NOT provide our numerical gains/sigmas; all "
            "numerical parameters are engineering/test parameters for this synthetic pipeline."
        ),
        "reference": reference.to_dict(master_seed),
        "search": search.to_dict(master_seed),
    }


# =============================================================================
# sem_intensity.py -- B2.7a experimental SE edge-response forward model
# =============================================================================

DEFAULT_EDGE_RESPONSE_STRENGTH = 0.0
DEFAULT_EDGE_RESPONSE_WIDTH_NM = 10.0
EDGE_RESPONSE_MODEL = "se_edge_response_geometry_gradient_peak"

LITERATURE_BASIS = (
    "Postek, Vladar, Villarrubia & Muto (2016), 'Comparison of Electron Imaging Modes for "
    "Dimensional Measurements in the Scanning Electron Microscope' (Microscopy and "
    "Microanalysis): SE signal is strongly affected by local feature orientation/topography, "
    "increases markedly at highly sloped feature edges/sidewalls, and is broadened by finite "
    "beam size and interaction volume. The paper does NOT specify our numerical grayscale "
    "amplitude or edge width; edge_response_strength and edge_response_width_nm are "
    "ENGINEERING EXPERIMENTAL PARAMETERS."
)


def build_edge_response_map(
    high_res: np.ndarray, edge_response_width_nm: float = DEFAULT_EDGE_RESPONSE_WIDTH_NM, nm_per_pixel: float = 1.0,
) -> np.ndarray:
    f = high_res.astype(np.float64)
    edge = np.zeros(f.shape, dtype=bool)
    edge[1:, :] |= f[1:, :] != f[:-1, :]
    edge[:-1, :] |= f[1:, :] != f[:-1, :]
    edge[:, 1:] |= f[:, 1:] != f[:, :-1]
    edge[:, :-1] |= f[:, 1:] != f[:, :-1]
    edge = edge.astype(np.float64)

    sigma_px = max(float(edge_response_width_nm), 0.0) / float(nm_per_pixel)
    if sigma_px > 0.0:
        broadened = ndi.gaussian_filter(edge, sigma=sigma_px, mode="nearest")
    else:
        broadened = edge
    return broadened


def apply_se_edge_response(
    high_res: np.ndarray, *, edge_response_strength: float = DEFAULT_EDGE_RESPONSE_STRENGTH,
    edge_response_width_nm: float = DEFAULT_EDGE_RESPONSE_WIDTH_NM, nm_per_pixel: float = 1.0,
) -> np.ndarray:
    f = high_res.astype(np.float64)
    if edge_response_strength <= 0.0:
        return f
    response = build_edge_response_map(
        high_res, edge_response_width_nm=edge_response_width_nm, nm_per_pixel=nm_per_pixel,
    )
    return f + edge_response_strength * response


def build_sem_edge_response_metadata(edge_response_strength: float, edge_response_width_nm: float) -> dict | str:
    if edge_response_strength <= 0.0:
        return "not_applied_b27a"
    return {
        "enabled": True,
        "model": EDGE_RESPONSE_MODEL,
        "applied_before": "gaussian_psf_and_b26a_b26b_noise",
        "applied_to": "shared_high_resolution_physical_master_1nm_px",
        "derived_from": "geometry_feature_boundaries_gradient_not_random_noise",
        "edge_response_strength": float(edge_response_strength),
        "edge_response_width_nm": float(edge_response_width_nm),
        "normalization": (
            "response = strength * gaussian_filter(boundary_map) with a unit-integral "
            "Gaussian kernel; each boundary pixel contributes total integrated intensity = "
            "strength, so dense-array edges gain roughly 0.2-0.3 x strength"
        ),
        "engineering_parameters": True,
        "literature_basis": LITERATURE_BASIS,
    }


# =============================================================================
# generate_pair.py -- reference/search pair assembly
# =============================================================================

REFERENCE_SIZE_PX = 1_000
SEARCH_SIZE_PX = 1_000
REFERENCE_NM_PER_PIXEL = 1
SEARCH_NM_PER_PIXEL = 10
REFERENCE_FOV_NM = REFERENCE_SIZE_PX * REFERENCE_NM_PER_PIXEL
SEARCH_FOV_NM = SEARCH_SIZE_PX * SEARCH_NM_PER_PIXEL


def difficulty_parameters(name: str) -> dict:
    """Return generate_sample-compatible kwargs for a B3.4 difficulty level."""
    if name not in DIFFICULTY_LEVELS:
        raise ValueError(f"unknown difficulty level {name!r}; expected one of {sorted(DIFFICULTY_LEVELS)}")
    c = DIFFICULTY_LEVELS[name]
    return {
        "amplitude_nm": c.amplitude_nm,
        "psf_sigma_nm": c.psf_sigma_nm,
        "noise_enabled": c.noise_enabled,
        "line_scan_correlation_strength": c.line_scan_correlation_strength,
        "cdu_dw_wl_nm": c.cdu_dw_wl_nm,
        "cdu_dw_bl_nm": c.cdu_dw_bl_nm,
        "cdu_ds_blc_nm": c.cdu_ds_blc_nm,
        "cdu_ds_snc_nm": c.cdu_ds_snc_nm,
        "overlay_dx_snc_nm": c.overlay_dx_snc_nm,
        "overlay_dy_snc_nm": c.overlay_dy_snc_nm,
    }


def target_for_seed(seed: int) -> tuple[int, int]:
    """Return a deterministic physical target inside the DRAM array regions."""
    if seed == 2026:
        return 1_700, 1_700

    master = build_clean_master(seed)
    regions = master["array_regions"]
    s_reg = _deterministic_seed(seed, "target_region")
    reg = regions[s_reg % len(regions)]
    left, top, right, bottom = reg["bounds_nm"]

    x_min, x_max = left + 500, right - 500
    y_min, y_max = top + 500, bottom - 500

    sx = _deterministic_seed(seed, "target_x")
    sy = _deterministic_seed(seed, "target_y")

    tx = x_min + (sx % ((x_max - x_min) // 10 + 1)) * 10
    ty = y_min + (sy % ((y_max - y_min) // 10 + 1)) * 10
    return int(tx), int(ty)


def generate_sample(
    seed: int = 2026,
    reference_path: Path | None = None,
    search_path: Path | None = None,
    metadata_path: Path | None = None,
    amplitude_nm: float = DEFAULT_VARIATION_AMPLITUDE_NM,
    correlation_length_nm: float = DEFAULT_CORRELATION_LENGTH_NM,
    psf_sigma_nm: float = DEFAULT_PSF_SIGMA_NM,
    noise_enabled: bool = False,
    reference_gain_e_per_intensity: float = DEFAULT_REFERENCE_NOISE.gain_e_per_intensity,
    reference_detector_sigma_intensity: float = DEFAULT_REFERENCE_NOISE.detector_sigma_intensity,
    search_gain_e_per_intensity: float = DEFAULT_SEARCH_NOISE.gain_e_per_intensity,
    search_detector_sigma_intensity: float = DEFAULT_SEARCH_NOISE.detector_sigma_intensity,
    line_scan_correlation_strength: float = 0.0,
    reference_line_scan_correlation_length_px: float = DEFAULT_REFERENCE_LINE_SCAN_CORRELATION_LENGTH_PX,
    search_line_scan_correlation_length_px: float = DEFAULT_SEARCH_LINE_SCAN_CORRELATION_LENGTH_PX,
    scan_axis: int = DEFAULT_SCAN_AXIS,
    edge_response_strength: float = DEFAULT_EDGE_RESPONSE_STRENGTH,
    edge_response_width_nm: float = DEFAULT_EDGE_RESPONSE_WIDTH_NM,
    cdu_dw_wl_nm: float = DEFAULT_CDU_DW_WL_NM,
    cdu_dw_bl_nm: float = DEFAULT_CDU_DW_BL_NM,
    cdu_ds_blc_nm: float = DEFAULT_CDU_DS_BLC_NM,
    cdu_ds_snc_nm: float = DEFAULT_CDU_DS_SNC_NM,
    overlay_dx_snc_nm: float = DEFAULT_OVERLAY_DX_SNC_NM,
    overlay_dy_snc_nm: float = DEFAULT_OVERLAY_DY_SNC_NM,
    parameter_engine: dict | None = None,
) -> tuple[Path, Path, Path]:
    """Write the reference/search sample pair and reproducible metadata.

    Same generation math as the original generate_pair.py: for the same
    (seed, knob) inputs the pixels are byte-identical. The only difference
    from the original is where the outputs land -- ``reference_path`` and
    ``search_path`` are required explicit file paths (matching the
    ``--reference``/``--search`` convention used by localize.py) instead of
    a fixed output_dir with hardcoded filenames.
    """
    if reference_path is None or search_path is None:
        raise ValueError("reference_path and search_path are required")
    reference_path = Path(reference_path)
    search_path = Path(search_path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    search_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_path) if metadata_path is not None else reference_path.parent / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    master = build_clean_master(seed=seed)
    target_x_nm, target_y_nm = target_for_seed(seed)
    reference_origin_x_nm = target_x_nm - REFERENCE_FOV_NM // 2
    reference_origin_y_nm = target_y_nm - REFERENCE_FOV_NM // 2

    # 1. Render ONE high-resolution 1 nm/px master with correlated edge variation.
    raw_master = render_high_res_master(
        master,
        amplitude_nm=amplitude_nm,
        correlation_length_nm=correlation_length_nm,
        seed=seed,
        cdu_dw_wl_nm=cdu_dw_wl_nm,
        cdu_dw_bl_nm=cdu_dw_bl_nm,
        cdu_ds_blc_nm=cdu_ds_blc_nm,
        cdu_ds_snc_nm=cdu_ds_snc_nm,
        overlay_dx_snc_nm=overlay_dx_snc_nm,
        overlay_dy_snc_nm=overlay_dy_snc_nm,
    )

    # B2.7a (experimental): optional SE-like edge response.
    intensity_master = apply_se_edge_response(
        raw_master,
        edge_response_strength=edge_response_strength,
        edge_response_width_nm=edge_response_width_nm,
        nm_per_pixel=REFERENCE_NM_PER_PIXEL,
    )

    # 2. Gaussian PSF on the (optionally edge-enhanced) high-resolution master.
    psf_master = apply_gaussian_psf(intensity_master, sigma_nm=psf_sigma_nm, nm_per_pixel=REFERENCE_NM_PER_PIXEL)

    # 2. Reference: exact 1000x1000 crop at 1 nm/px from the SAME PSF master.
    ref_crop = psf_master[
        reference_origin_y_nm: reference_origin_y_nm + REFERENCE_FOV_NM,
        reference_origin_x_nm: reference_origin_x_nm + REFERENCE_FOV_NM,
    ]

    # 3. Search: 10x10 area averaging from the SAME PSF master -> 1000x1000 at 10 nm/px.
    downsample_factor = SEARCH_NM_PER_PIXEL // REFERENCE_NM_PER_PIXEL
    search_array = area_downsample(psf_master, factor=downsample_factor)

    # B2.6a: independent Poisson-Gaussian noise AFTER deterministic formation.
    reference_noise = NoiseParameters(
        enabled=noise_enabled, model="poisson_gaussian", stream_id="reference",
        gain_e_per_intensity=reference_gain_e_per_intensity,
        detector_sigma_intensity=reference_detector_sigma_intensity,
    )
    search_noise = NoiseParameters(
        enabled=noise_enabled, model="poisson_gaussian", stream_id="search",
        gain_e_per_intensity=search_gain_e_per_intensity,
        detector_sigma_intensity=search_detector_sigma_intensity,
    )
    ref_final = ref_crop
    search_final = search_array
    if noise_enabled:
        ref_final = apply_sem_noise(
            ref_crop, seed=noise_stream_seed(seed, "reference"),
            gain_e_per_intensity=reference_gain_e_per_intensity,
            detector_sigma_intensity=reference_detector_sigma_intensity,
            line_scan_correlation_strength=line_scan_correlation_strength,
            line_scan_correlation_length_px=reference_line_scan_correlation_length_px,
            scan_axis=scan_axis,
        )
        search_final = apply_sem_noise(
            search_array, seed=noise_stream_seed(seed, "search"),
            gain_e_per_intensity=search_gain_e_per_intensity,
            detector_sigma_intensity=search_detector_sigma_intensity,
            line_scan_correlation_strength=line_scan_correlation_strength,
            line_scan_correlation_length_px=search_line_scan_correlation_length_px,
            scan_axis=scan_axis,
        )
    reference = Image.fromarray(np.clip(np.round(ref_final), 0, 255).astype(np.uint8), mode="L")
    search = Image.fromarray(np.clip(np.round(search_final), 0, 255).astype(np.uint8), mode="L")

    gt_x = target_x_nm / SEARCH_NM_PER_PIXEL
    gt_y = target_y_nm / SEARCH_NM_PER_PIXEL
    metadata = {
        "seed": seed,
        "master_size_nm": master["master_size_nm"],
        "reference_size_px": [REFERENCE_SIZE_PX, REFERENCE_SIZE_PX],
        "search_size_px": [SEARCH_SIZE_PX, SEARCH_SIZE_PX],
        "reference_magnification_x": REFERENCE_MAGNIFICATION_X,
        "search_magnification_x": SEARCH_MAGNIFICATION_X,
        "reference_nm_per_pixel": REFERENCE_NM_PER_PIXEL,
        "search_nm_per_pixel": SEARCH_NM_PER_PIXEL,
        "reference_fov_nm": [REFERENCE_FOV_NM, REFERENCE_FOV_NM],
        "search_fov_nm": [SEARCH_FOV_NM, SEARCH_FOV_NM],
        "target_x_nm": target_x_nm,
        "target_y_nm": target_y_nm,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "coordinate_convention": "continuous zero-based pixel-edge coordinates: (0,0) is the upper-left outer image edge; pixel (column,row) covers [column,column+1) x [row,row+1), and physical nm maps by division by nm_per_pixel",
        "reference_crop_origin_nm": [reference_origin_x_nm, reference_origin_y_nm],
        "reference_crop_center_nm": [target_x_nm, target_y_nm],
        "search_origin_nm": [0, 0],
        "intensity_mapping": INTENSITY_MAPPING,
        "physical_source": "B2.1 parametric_hierarchical_rectangles",
        "physical_variation": {
            "enabled": amplitude_nm > 0.0,
            "variation_model": "correlated_edge_displacement_ler_lwr",
            "amplitude_nm": float(amplitude_nm),
            "correlation_length_nm": float(correlation_length_nm),
            "literature_basis": (
                "ITRS (2015) More Moore / IRDS (2017, 2024) More Moore justify the physical "
                "existence and critical importance of CDU/LER/LWR variation in nanoscale devices. "
                "Numerical amplitude and correlation length are engineering sweep parameters."
            ),
        },
        "psf_enabled": psf_sigma_nm > 0.0,
        "psf_model": "gaussian_2d",
        "psf_sigma_nm": float(psf_sigma_nm),
        "high_resolution_nm_per_pixel": REFERENCE_NM_PER_PIXEL,
        "search_downsampling_method": "area_averaging_10x10",
        "psf_literature_basis": (
            "Kamal & Hailstone (2024), 'Determination of the Optical PSF in the Scanning "
            "Electron Microscope Using Single Nanoparticle Images' (justifies finite SEM probe/PSF). "
            "Gaussian is an engineering approximation; sigma_nm is an engineering experimental "
            "parameter and is not claimed to be a universal literature-derived value."
        ),
        "layout_sha256": master["layout_sha256"],
    }

    if edge_response_strength > 0.0:
        metadata["sem_edge_response"] = build_sem_edge_response_metadata(edge_response_strength, edge_response_width_nm)

    if any(v != 0.0 for v in (cdu_dw_wl_nm, cdu_dw_bl_nm, cdu_ds_blc_nm, cdu_ds_snc_nm)):
        metadata["cdu"] = build_cdu_metadata(cdu_dw_wl_nm, cdu_dw_bl_nm, cdu_ds_blc_nm, cdu_ds_snc_nm)

    if overlay_dx_snc_nm != 0.0 or overlay_dy_snc_nm != 0.0:
        metadata["overlay"] = build_overlay_metadata(overlay_dx_snc_nm, overlay_dy_snc_nm)

    if noise_enabled:
        metadata["noise"] = build_noise_metadata(
            seed, reference_noise, search_noise,
            line_scan_correlation_strength=line_scan_correlation_strength,
            reference_line_scan_correlation_length_px=reference_line_scan_correlation_length_px,
            search_line_scan_correlation_length_px=search_line_scan_correlation_length_px,
            scan_axis=scan_axis,
        )

    if parameter_engine is not None:
        metadata["parameter_engine"] = parameter_engine

    reference.save(reference_path)
    search.save(search_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return reference_path, search_path, metadata_path


# =============================================================================
# CLI -- --reference / --search output paths (matches localize.py's style)
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reference", "-r", type=Path, default=Path("reference.png"), help="Output path for reference.png (1000x1000, 1 nm/px). Ignored when --count > 1.")
    parser.add_argument("--search", "-s", type=Path, default=Path("search.png"), help="Output path for search.png (1000x1000, 10 nm/px). Ignored when --count > 1.")
    parser.add_argument("--metadata", "-m", type=Path, default=None, help="Output path for metadata.json (defaults next to --reference). Ignored when --count > 1.")
    parser.add_argument("--count", "-n", type=int, default=1, help="Generate this many pairs instead of one. Each pair gets its own numbered subfolder under --output-dir and a distinct seed (--seed, --seed+1, ...).")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("."), help="Parent directory for pair_001/, pair_002/, ... when --count > 1")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for a single pair, or the starting seed when --count > 1")
    parser.add_argument(
        "--difficulty", choices=sorted(DIFFICULTY_LEVELS), default=None,
        help="B3.4 controlled difficulty level (L0/L1/L2/L3); explicit per-flag values below override the level for that knob",
    )
    parser.add_argument("--amplitude", type=float, default=DEFAULT_VARIATION_AMPLITUDE_NM, help="Physical variation amplitude in nm (0=clean baseline)")
    parser.add_argument("--correlation-length", type=float, default=DEFAULT_CORRELATION_LENGTH_NM, help="Physical correlation length in nm")
    parser.add_argument("--sigma", type=float, default=DEFAULT_PSF_SIGMA_NM, help="Gaussian PSF sigma in nm")
    parser.add_argument("--noise", action="store_true", help="Enable B2.6a Poisson-Gaussian SEM noise")
    parser.add_argument("--ref-gain", type=float, default=DEFAULT_REFERENCE_NOISE.gain_e_per_intensity)
    parser.add_argument("--ref-detector-sigma", type=float, default=DEFAULT_REFERENCE_NOISE.detector_sigma_intensity)
    parser.add_argument("--search-gain", type=float, default=DEFAULT_SEARCH_NOISE.gain_e_per_intensity)
    parser.add_argument("--search-detector-sigma", type=float, default=DEFAULT_SEARCH_NOISE.detector_sigma_intensity)
    parser.add_argument("--line-scan-correlation", type=float, default=0.0, help="B2.6b line-scan correlation strength in [0,1]")
    parser.add_argument("--ref-line-scan-length", type=float, default=DEFAULT_REFERENCE_LINE_SCAN_CORRELATION_LENGTH_PX)
    parser.add_argument("--search-line-scan-length", type=float, default=DEFAULT_SEARCH_LINE_SCAN_CORRELATION_LENGTH_PX)
    parser.add_argument("--edge-response-strength", type=float, default=DEFAULT_EDGE_RESPONSE_STRENGTH)
    parser.add_argument("--edge-response-width", type=float, default=DEFAULT_EDGE_RESPONSE_WIDTH_NM)
    parser.add_argument("--cdu-dw-wl", type=float, default=DEFAULT_CDU_DW_WL_NM)
    parser.add_argument("--cdu-dw-bl", type=float, default=DEFAULT_CDU_DW_BL_NM)
    parser.add_argument("--cdu-ds-blc", type=float, default=DEFAULT_CDU_DS_BLC_NM)
    parser.add_argument("--cdu-ds-snc", type=float, default=DEFAULT_CDU_DS_SNC_NM)
    parser.add_argument("--overlay-dx-snc", type=float, default=DEFAULT_OVERLAY_DX_SNC_NM)
    parser.add_argument("--overlay-dy-snc", type=float, default=DEFAULT_OVERLAY_DY_SNC_NM)
    args = parser.parse_args()

    # --difficulty expands into the covered knobs; an explicitly-passed flag
    # (differing from its own CLI default) overrides the level for that knob.
    level_kwargs = difficulty_parameters(args.difficulty) if args.difficulty else {}
    flag_defaults = {
        "amplitude": DEFAULT_VARIATION_AMPLITUDE_NM,
        "sigma": DEFAULT_PSF_SIGMA_NM,
        "noise": False,
        "line_scan_correlation": 0.0,
        "cdu_dw_wl": DEFAULT_CDU_DW_WL_NM,
        "cdu_dw_bl": DEFAULT_CDU_DW_BL_NM,
        "cdu_ds_blc": DEFAULT_CDU_DS_BLC_NM,
        "cdu_ds_snc": DEFAULT_CDU_DS_SNC_NM,
        "overlay_dx_snc": DEFAULT_OVERLAY_DX_SNC_NM,
        "overlay_dy_snc": DEFAULT_OVERLAY_DY_SNC_NM,
    }
    explicit = {
        "amplitude_nm": args.amplitude != flag_defaults["amplitude"],
        "psf_sigma_nm": args.sigma != flag_defaults["sigma"],
        "noise_enabled": args.noise,
        "line_scan_correlation_strength": args.line_scan_correlation != flag_defaults["line_scan_correlation"],
        "cdu_dw_wl_nm": args.cdu_dw_wl != flag_defaults["cdu_dw_wl"],
        "cdu_dw_bl_nm": args.cdu_dw_bl != flag_defaults["cdu_dw_bl"],
        "cdu_ds_blc_nm": args.cdu_ds_blc != flag_defaults["cdu_ds_blc"],
        "cdu_ds_snc_nm": args.cdu_ds_snc != flag_defaults["cdu_ds_snc"],
        "overlay_dx_snc_nm": args.overlay_dx_snc != flag_defaults["overlay_dx_snc"],
        "overlay_dy_snc_nm": args.overlay_dy_snc != flag_defaults["overlay_dy_snc"],
    }
    values = {
        "amplitude_nm": args.amplitude,
        "psf_sigma_nm": args.sigma,
        "noise_enabled": args.noise,
        "line_scan_correlation_strength": args.line_scan_correlation,
        "cdu_dw_wl_nm": args.cdu_dw_wl,
        "cdu_dw_bl_nm": args.cdu_dw_bl,
        "cdu_ds_blc_nm": args.cdu_ds_blc,
        "cdu_ds_snc_nm": args.cdu_ds_snc,
        "overlay_dx_snc_nm": args.overlay_dx_snc,
        "overlay_dy_snc_nm": args.overlay_dy_snc,
    }
    for key, val in values.items():
        if explicit[key]:
            level_kwargs[key] = val

    common_kwargs = dict(
        correlation_length_nm=args.correlation_length,
        reference_gain_e_per_intensity=args.ref_gain,
        reference_detector_sigma_intensity=args.ref_detector_sigma,
        search_gain_e_per_intensity=args.search_gain,
        search_detector_sigma_intensity=args.search_detector_sigma,
        reference_line_scan_correlation_length_px=args.ref_line_scan_length,
        search_line_scan_correlation_length_px=args.search_line_scan_length,
        edge_response_strength=args.edge_response_strength,
        edge_response_width_nm=args.edge_response_width,
        **level_kwargs,
    )

    if args.count <= 1:
        reference_path, search_path, metadata_path = generate_sample(
            seed=args.seed,
            reference_path=args.reference,
            search_path=args.search,
            metadata_path=args.metadata,
            **common_kwargs,
        )
        print(f"Wrote {reference_path}")
        print(f"Wrote {search_path}")
        print(f"Wrote {metadata_path}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pad = max(3, len(str(args.count)))
        for i in range(args.count):
            seed_i = args.seed + i
            pair_dir = args.output_dir / f"pair_{i + 1:0{pad}d}"
            reference_path, search_path, metadata_path = generate_sample(
                seed=seed_i,
                reference_path=pair_dir / "reference.png",
                search_path=pair_dir / "search.png",
                metadata_path=pair_dir / "metadata.json",
                **common_kwargs,
            )
            print(f"[{i + 1}/{args.count}] seed={seed_i} -> {pair_dir}")


if __name__ == "__main__":
    main()
