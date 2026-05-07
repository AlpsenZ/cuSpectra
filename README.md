# CUDA-Accelerated Molecular Absorption Spectroscopy

GPU-accelerated computation of molecular absorption spectra using NVIDIA CUDA, featuring two complementary approaches: a rigorous line-by-line (LBL) method and an ultrafast spectral reconstruction method (USAR) based on pre-fitted polynomial surface models.

## Overview

This project provides high-performance tools for computing molecular absorption coefficients and transmittance spectra, with a focus on CO₂ infrared spectroscopy. All heavy computation is offloaded to the GPU using Numba CUDA, achieving significant speedups over CPU-based implementations.

The project is organized into two independent modules:

### 1. cuLBL — CUDA Line-by-Line Method

A rigorous Voigt-profile-based line-by-line radiative transfer computation. Each spectral line is convolved with a Voigt profile (computed via the Humlíček algorithm) and accumulated onto a wavenumber grid. The implementation supports:

- Temperature and pressure correction of line strengths (via HITRAN partition sums)
- Doppler and Lorentz broadening (half-width at half-maximum)
- Voigt profile evaluation using the Humlíček rational approximation with multiple asymptotic regions
- Grid-strided CUDA kernel launch for optimal GPU occupancy
- Absorption coefficient database generation over P/T grids

### 2. cuUSAR — Ultrafast Spectral Absorption Reconstruction

A fast spectral reconstruction method that uses pre-fitted 5th-order polynomial surfaces `k(ν; P, T)` to bypass on-the-fly line-by-line computation. Each wavenumber channel is modeled as a piecewise bivariate polynomial over four P/T sub-regions, enabling rapid spectrum generation for arbitrary (P, T) within the calibrated range (0.1–2 atm, 250–1250 K).

## File Structure

### LBL Module

| File | Description |
|------|-------------|
| `lbl_verify.py` | Verification run: computes LBL absorption coefficients at a specific (P, T) condition and saves results |
| `lbl_transmittance.py` | Computes LBL absorption spectrum and transmittance, used for result comparison |
| `lbl_benchmark_pt_outer.py` | GPU kernel parameter benchmark — sampling-point-outer parallelization strategy |
| `lbl_benchmark_nu_outer.py` | GPU kernel parameter benchmark — wavenumber-outer parallelization strategy (with atomicAdd) |
| `lbl_build_database.py` | Builds an absorption coefficient database over a regular (P, T) grid (0.1–2 atm, 250–1250 K) |
| `lbl_calc_path_specific.py` | Computes LBL absorption coefficients on a specific hardcoded (P, T) path |
| `lbl_calc_path_from_file.py` | Computes LBL absorption coefficients on a (P, T) path loaded from an external .mat file |

### USAR Module

| File | Description |
|------|-------------|
| `usar_fastspec_wn_outer.py` | Fast spectrum reconstruction parallelized by wavenumber channels |
| `usar_fastspec_pt_outer.py` | Fast spectrum reconstruction parallelized by P/T sample points |
| `usar_benchmark_pt_outer.py` | GPU kernel parameter benchmark for PT-outer parallelization variant |
| `usar_benchmark_wn_outer.py` | GPU kernel parameter benchmark for wavenumber-outer parallelization variant |
| `usar_calc_path.py` | Computes reconstructed spectra on a specific (P, T) path |

## Technical Architecture

### Physical Model

The absorption coefficient at wavenumber ν is computed as:

```
k(ν) = K · Σ S_j(T) · V(ν - ν_j, γ_D, γ_L)
```

where:
- **K** — volume concentration factor
- **S_j(T)** — temperature-corrected line strength for line j
- **V** — Voigt profile (convolution of Doppler and Lorentz profiles)
- **γ_D** — Doppler HWHM, proportional to ν√(T/M)
- **γ_L** — Lorentz HWHM, with temperature exponent from HITRAN

### Voigt Profile (Humlíček Algorithm)

The Voigt function is evaluated using Humlíček's rational approximation (CPF12 / W4 algorithm), which partitions the (x, y) parameter space into four regions for optimal accuracy and performance:

- **Region I**: `|x| + y > 15` — asymptotic far-wing expansion (2nd/4th order)
- **Region II**: `5.5 < |x| + y ≤ 15` — intermediate region (8th order rational)
- **Region III**: `|x| + y ≤ 5.5` and `y > 0.195|x| - 0.176` — near-Lorentz region (10th order)
- **Region IV**: `|x| + y ≤ 5.5` and `y ≤ 0.195|x| - 0.176` — near-Doppler/Gaussian region (28th order)

### USAR Surface Model

The ultrafast method fits a 5th-order bivariate polynomial:

```
k(ν; P, T) = Σ_i Σ_j c_ij · (P - μ_P)^i · (T - μ_T)^j
```

The (P, T) domain (0.1–2 atm, 250–1250 K) is divided into 4 sub-regions, each with its own set of 21 polynomial coefficients per wavenumber channel. Parameters are normalized per region for numerical stability.

### CUDA Parallelization Strategies

Two grid-stride loop patterns are explored:

1. **Wavenumber-outer**: Each CUDA thread processes one wavenumber channel, iterating over all spectral lines or P/T points in the inner loop
2. **Sample-point-outer**: Each CUDA thread processes one P/T sample point or spectral line, iterating over wavenumber channels in the inner loop

GPU kernel parameters (threads per block, grid stride) are benchmarked to find optimal configurations for different hardware.

## Requirements

### Software Dependencies

- Python 3.8+
- NumPy
- SciPy
- Numba (with CUDA support)
- Matplotlib
- HAPI (HITRAN Application Programming Interface)

### Hardware Requirements

- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- CUDA Toolkit 10.0+ (for Numba CUDA JIT compilation)

### Installation

```bash
# Install core dependencies
pip install numpy scipy numba matplotlib

# Install HAPI
pip install hitran-api
```

> **Important**: HAPI and HITRAN databases must be downloaded separately from [hitran.org](https://hitran.org).

## HITRAN Database Setup

**The HAPI and HITRAN spectral line databases are NOT included in this repository.** Users must:

1. Visit [https://hitran.org](https://hitran.org) to register and download the required HITRAN line-by-line data
2. Use HAPI's `fetch()` or `db_begin()` functions to download and manage the required molecular line lists

The default database directory expected by the scripts is `./data/`. You can configure this in HAPI:

```python
from hapi import db_begin
db_begin('data')  # Use your local HITRAN data directory
```

The scripts use the following HITRAN tables (fetched via HAPI):

- `CO2_1900_3500` — CO₂ lines in the 1900–3500 cm⁻¹ range
- `CO2_1000_11000` — CO₂ lines in the 1000–11000 cm⁻¹ range

To download these tables via HAPI:

```python
from hapi import fetch, db_begin

db_begin('data')
fetch('CO2', 2, 1, 1900, 3500)    # CO2, isotopologue 1, 1900-3500 cm⁻¹
fetch('CO2', 2, 1, 1000, 11000)   # CO2, isotopologue 1, 1000-11000 cm⁻¹
```

## Usage Examples

### LBL Verification Run

```python
python lbl_verify.py
```

Computes CO₂ absorption coefficients at P ≈ 622 K, T ≈ 1.9 atm from the `CO2_1900_3500` table. Results are saved as `lbl_verify_results.mat`.

### Build Absorption Coefficient Database

```python
python lbl_build_database.py
```

Generates absorption coefficient data over a P/T grid (P: 0.1–2.0 atm, step 0.1 atm; T: 250–1250 K, step 5 K). Results are saved per pressure level as `.mat` files.

### USAR Fast Spectrum Reconstruction

```python
python usar_fastspec_wn_outer.py
```

Loads pre-fitted surface parameters from `a2509_cuUSAR_paras_nonuniform.mat` and generates spectra for 1000 random (P, T) points. Requires the surface parameter file to be present.

### GPU Kernel Benchmarking

```python
python lbl_benchmark_nu_outer.py
```

Tests combinations of threads-per-block (32–1024) and grid-stride (1–6) to find the optimal CUDA launch configuration for your GPU.

## Configuration

Key parameters adjustable in each script:

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `wave_start` | Lower wavenumber bound (cm⁻¹) | 2000 |
| `wave_end` | Upper wavenumber bound (cm⁻¹) | 3333 or 10000 |
| `sampling_delta` | Spectral resolution (cm⁻¹) | 0.01 to 0.0005 |
| `wing_cut` | Line wing cutoff (cm⁻¹) | 25 |
| `molec_id` | HITRAN molecule ID (2 = CO₂) | 2 |
| `local_iso_id` | Isotopologue ID (1 = main) | 1 |

## Output Format

All output data is saved in MATLAB `.mat` format using SciPy's `savemat`. Key output variables include:

- `Coef` / `coef` — Absorption coefficient arrays (cm⁻¹)
- `Pc` / `Px` — Pressure values (atm)
- `Tc` / `Tx` — Temperature values (K)
- `nu` — Wavenumber grid (cm⁻¹)
- `trans` — Transmittance spectrum (unitless)
- `results` — GPU kernel benchmark timing results

## Limitations & Notes

- The USAR method uses pre-fitted parameter files (e.g., `a2509_cuUSAR_paras_nonuniform.mat`) which must be generated beforehand using the LBL database builder and a surface fitting procedure (not included in this repository)
- P/T extrapolation beyond the calibration range (0.1–2 atm, 250–1250 K) will produce incorrect results; the USAR kernel prints a warning and returns 0
- Currently hardcoded for CO₂ (molecule ID 2, isotopologue 1); modification of `molec_id`, `local_iso_id`, and `table_name` is required for other species
- The `absor_coef` parameter in LBL kernels serves as a zero-initialization placeholder — it must not be removed from the kernel signature
- Performance scales with spectral line count, sampling resolution, and wing cutoff distance

## License

This project is provided as open-source. Please include attribution when using or modifying this code.

## References

- HITRAN Database: [https://hitran.org](https://hitran.org)
- HAPI Documentation: [https://hitran.org/hapi](https://hitran.org/hapi)
- Humlíček, J. (1982). "Optimized computation of the Voigt and complex probability functions." *J. Quant. Spectrosc. Radiat. Transfer*, 27(4), 437–444.
- Numba CUDA Documentation: [https://numba.readthedocs.io/en/stable/cuda/index.html](https://numba.readthedocs.io/en/stable/cuda/index.html)
