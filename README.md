# Drift-Sense

**ML-Powered SEM Pattern Localization for Semiconductor Metrology**  
*SEMICON India Hackathon 2026 Submission | Target: High-Density DRAM Architecture (6F² Physical Pitch)*

Drift-Sense is an end-to-end Machine Learning pipeline that accurately localizes high-magnification reference SEM templates ($1\,\text{nm/px}$, $1000\times1000$ region) within wide field-of-view search overviews ($10\,\text{nm/px}$, $1000\times1000$ image). By combining FFT-based Sum of Squared Differences (SSD) candidate generation with a frozen 20-feature Tuned Random Forest classifier, Drift-Sense resolves dense periodic lattice ambiguity and SEM line-scan noise without requiring metadata or ground truth during inference.

---

## Quick Start (Shortest Reviewer Workflow)

Copy and paste these commands into your terminal for an immediate end-to-end run:

### Windows (PowerShell)
```powershell
# 1. Enter submission directory
cd FINAL_SUBMISSION\Drift-Sense

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install numpy scipy Pillow scikit-learn joblib

# 4. Generate one sample DRAM image pair (L3 Full Physical Model)
python generate_pair_standalone.py --reference sample_pair/reference.png --search sample_pair/search.png --metadata sample_pair/metadata.json --difficulty L3 --seed 2026

# 5. Run localization inference
python localize.py --reference sample_pair/reference.png --search sample_pair/search.png
```

### Linux / macOS (Bash)
```bash
# 1. Enter submission directory
cd FINAL_SUBMISSION/Drift-Sense

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install numpy scipy Pillow scikit-learn joblib

# 4. Generate one sample DRAM image pair (L3 Full Physical Model)
python generate_pair_standalone.py --reference sample_pair/reference.png --search sample_pair/search.png --metadata sample_pair/metadata.json --difficulty L3 --seed 2026

# 5. Run localization inference
python localize.py --reference sample_pair/reference.png --search sample_pair/search.png
```

---

## Step-by-Step Execution Guide

```
  1. CLONE & SET UP ENVIRONMENT
                 ↓
  2. INSTALL DEPENDENCIES
                 ↓
  3. GENERATE SAMPLE IMAGE PAIR  ──> [reference.png, search.png, metadata.json]
                 ↓
  4. RUN LOCALIZATION INFERENCE  ──> localize.py (loads model.joblib)
                 ↓
  5. RECEIVE PREDICTED (X, Y)
                 ↓
  6. (OPTIONAL) VERIFY VS METADATA
                 ↓
  7. (OPTIONAL) RUN BATCH TEST
```

---

### Step 1: Prerequisites & Virtual Environment

- **Python Version:** Python 3.10, 3.11, 3.12, or 3.13
- **Supported Platforms:** Windows, Linux, macOS
- **Required Libraries:** `numpy`, `scipy`, `Pillow`, `scikit-learn`, `joblib`

Create and activate an isolated virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install numpy scipy Pillow scikit-learn joblib
```

---

### Step 3: Generate a Sample Image Pair

Generate a synthetic semiconductor DRAM image pair using [`generate_pair_standalone.py`](generate_pair_standalone.py):

```powershell
python generate_pair_standalone.py --reference test_pair/reference.png --search test_pair/search.png --metadata test_pair/metadata.json --difficulty L3 --seed 12345
```

> **Note:** Target folders (e.g., `test_pair/`) are created automatically.

#### Generated Files:
1. `reference.png` ($1000\times1000$, $1\,\text{nm/px}$): High-magnification template crop of the DRAM active array.
2. `search.png` ($1000\times1000$, $10\,\text{nm/px}$): Wide-overview search frame representing the $10\,\mu\text{m}\times10\,\mu\text{m}$ field of view.
3. `metadata.json`: Contains ground-truth coordinates (`gt_x`, `gt_y`) and physical parameters for post-inference verification.

#### Available Difficulty Levels (`--difficulty`):
- `L0`: Clean physical geometry baseline (no noise, no PSF).
- `L1`: Imaging effects (2D Gaussian probe PSF + SEM Poisson-Gaussian noise + line-scan correlation).
- `L2`: Process variation (L1 + CDU variation + layer overlay misregistration).
- `L3`: Full metrology model (L2 + correlated Line Edge Roughness).

---

### Step 4: Run Pattern Localization (`localize.py`)

Run the primary inference engine [`localize.py`](localize.py). 

> **Important:** `localize.py` takes **only** the image paths. It does **not** receive metadata or ground-truth coordinates.

```powershell
python localize.py --reference test_pair/reference.png --search test_pair/search.png
```

#### Standard Output Format:
```
=================================================================
  DRIFT-SENSE ML LOCALIZATION RESULT
=================================================================
  Predicted Center  : (575.0, 516.0)
  RF Confidence     : 0.9663
  Candidates Scored : 100
  Model             : RandomForestClassifier
  Runtime           : 1842.1 ms
=================================================================

{
  "predicted_x": 575.0,
  "predicted_y": 516.0,
  "confidence": 0.9663,
  "num_candidates": 100,
  "runtime_ms": 1842.1,
  "model_name": "RandomForestClassifier"
}
```

#### JSON-Only Output:
For automated test scripts, pass `--json`:
```powershell
python localize.py --reference test_pair/reference.png --search test_pair/search.png --json
```

---

### Step 5: Post-Inference Ground Truth Verification (Optional)

To check the accuracy of the prediction:
1. Open the generated `test_pair/metadata.json`.
2. Compare the `gt_x` and `gt_y` fields with `predicted_x` and `predicted_y`.
3. Compute Chebyshev error: $\max(|x_{\text{pred}} - x_{\text{gt}}|, |y_{\text{pred}} - y_{\text{gt}}|)$. A localization with error $\le 5.0\,\text{px}$ is considered a match.

---

### Step 6: Batch Generation & Testing (Optional)

To generate multiple test pairs in numbered subfolders:

```powershell
python generate_pair_standalone.py --count 30 --seed 100 --difficulty L3 --output-dir integration_test/data
```
This generates `integration_test/data/pair_001/` through `pair_030/`.

To run inference across all generated pairs in PowerShell:

```powershell
Get-ChildItem "integration_test\data" -Directory | ForEach-Object {
    Write-Host "`n===== $($_.Name) ====="
    python localize.py --reference "$($_.FullName)\reference.png" --search "$($_.FullName)\search.png"
}
```

---

## Two-Stage Architecture & Model Details

```
Reference Image (1nm/px)  ──┐
                            ├──> Area Downsampling (10x) -> 100x100 Template
Search Image   (10nm/px)  ──┘
                   │
                   ▼
       2D FFT SSD Correlation Map
                   │
                   ▼
     Top-100 Local Minima Candidates (5px Chebyshev Radius)
                   │
                   ▼
      20-Feature Statistical Matrix
                   │
                   ▼
    Tuned Random Forest Model (model.joblib, 200 Trees)
                   │
                   ▼
    Predicted Center Coordinate (X, Y)
```

### Model Resolution
- `localize.py` automatically resolves `model.joblib` located in its own directory (`Path(__file__).resolve().parent / "model.joblib"`).
- An explicit checkpoint path can be passed via `--model path/to/model.joblib`.
- The final model is the frozen **20-feature Tuned Random Forest Classifier** (200 trees, depth 15, balanced class weights).

### Training Reproduction (`C5_Localization_ML.ipynb`)
- Model training, validation, and diagnostic experiments are documented in [`C5_Localization_ML.ipynb`](C5_Localization_ML.ipynb).
- **No training is required for inference.** Pre-trained weights are packaged in `model.joblib`.

---

## Troubleshooting & FAQ

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: Trained model not found` | Command executed from another directory without `--model` | Run from `FINAL_SUBMISSION/Drift-Sense` or specify `--model FINAL_SUBMISSION/Drift-Sense/model.joblib` |
| `ModuleNotFoundError: No module named 'joblib'` | Dependencies not installed in active environment | Run `pip install joblib scikit-learn Pillow numpy scipy` |
| `FileNotFoundError: Reference image not found` | Typo in file path or file not yet generated | Verify path or run Step 3 to generate the pair |
| RF Confidence $<0.70$ | Extreme noise or boundary ambiguity | Indicates periodic ambiguity; model still outputs the argmax likelihood candidate |

---

## Scientific References

The physical and noise models implemented in Drift-Sense are based on established semiconductor metrology literature:

1. **SEM Noise & Line-Scan Physics:**  
   Roels et al. (2018), *"Bayesian Deconvolution of Scanning Electron Microscopy Images Using Point-spread Function Estimation and Non-local Regularization"*, IEEE Transactions on Image Processing.
2. **Secondary Electron Edge Response:**  
   Postek, M. T. et al. (2016), *"Comparison of Electron Imaging Modes for Dimensional Measurements in the Scanning Electron Microscope"*, Microscopy and Microanalysis.
3. **Finite Electron Beam PSF:**  
   Kamal, M. & Hailstone, R. (2024), *"Determination of the Optical PSF in the Scanning Electron Microscope Using Single Nanoparticle Images"*.
4. **Semiconductor Critical Dimension Uniformity (CDU) & Line Edge Roughness (LER):**  
   ITRS / IRDS More Moore Metrology Reports; Bunday et al., *CD-SEM Metrology for Nanoscale Devices*; Constantoudis et al., *Scaling and Correlation of LER in Nanopatterning*.
5. **Lithographic Overlay:**  
   Fukagawa et al., Sullivan & Shin, and Graff et al. (2023), *Nanometre Overlay and Alignment Metrology in Multi-Patterning Fabrication*.
