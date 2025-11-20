# PyRoddier - Project Context & Guidelines

## Project Overview

PyRoddier is a Python-based GUI application designed to perform the **Roddier Test** (Intra/Extra-focal curvature sensing) for telescope aberration analysis.
**CRITICAL GOAL:** Achieve numerical parity with **WinRoddier 3.0**. Currently, there is a discrepancy in the calculated Zernike coefficients/wavefront error compared to the reference software.

## Architecture

- **Language:** Python 3.x
- **GUI:** PyQt5
- **Math/Physics:** NumPy, SciPy (FFT, Zernike polynomials, Phase retrieval).
- **Visualization:** Matplotlib / PyQtGraph.

## Core Physics Workflow (The Roddier Test)

1. **Input:** Two images (Intra-focal and Extra-focal) taken at a distance $d$ from the focus.
2. **Preprocessing:** Centering, crop, and normalization of energy between images.
3. **Signal Calculation:** The normalized difference signal $S = (I_1 - I_2) / (I_1 + I_2)$.
4. **Phase Retrieval:** Solving the Poisson equation or using iterative FFT-based methods (Gerchberg-Saxton or similar) to reconstruct the wavefront phase from $S$.
5. **Zernike Decomposition:** Fitting the reconstructed wavefront to Zernike polynomials.

## Coding Guidelines

- **Style:** PEP8 compliant.
- **Type Hinting:** Use Python type hints for all function definitions, especially in the physics calculation modules.
- **Docstrings:** Explain the physical units (microns, arcseconds, pixels) in every math function.
- **Error Handling:** GUI must not crash on math errors (e.g., division by zero in background).

## Current Known Issues

- Results do not match WinRoddier 3.0.
- Suspected areas: FFT shifting, normalization factors, or Zernike radius scaling.
- Performance issues in the GUI when processing large images.

## Commands

- Run app: `python src/main.py` (o el archivo de entrada que tengas)
- Run tests: `pytest`

## Planned Features

The next features are inside the plan.md file.
