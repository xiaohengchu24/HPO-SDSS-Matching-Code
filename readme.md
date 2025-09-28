# Repository: HPO-SDSS-Matching-Code
# Research Code for: High-Probability Observable Artificial Objects from Historical Sloan Digital Sky Survey Data

## 1. Overview and Purpose
This repository contains the complete source code and configuration files used for the deep learning classification, orbital matching, and statistical analysis presented in the manuscript:   "High-Probability Observable Artificial Objects from Historical Sloan Digital Sky Survey Data"   ([2025]).

The code implements the three-stage filtering pipeline and the cross-correlation scheme detailed in the methods section of the paper. The goal is to:
1.  Identify satellite streak events from SDSS archival images using a custom   Swin Transformer   model.
2.  Cross-reference these streaks with Two-Line Element (TLE) orbital data.
3.  Derive the statistical characteristics of   High-Probability Observable (HPO)   celestial bodies.

## 2. Data Availability and Access

The primary input and final derived catalog data are archived to ensure permanent accessibility and reproducibility.

*   SDSS Raw Data:   Available via the official SDSS Data Release I and II archives.
*   Resulting Catalog (3,515 Matched Streaks):   The final matched catalog used for statistical analysis is provided in the `3_finally_match_result__include_filter_with_sdss_streak_information` directory. This data (split by SDSS filter: g, i, r, u, z) is the foundation for Figures 3, 4, and Table 1.
    *   Data DOI:   [ ]

## 3. Software Environment and Dependencies (Key to Reproducibility)

The code was developed and executed in a Python environment managed by   Conda   (named `xhc_v2.5`). All dependencies are listed precisely in the `environment.yml` file.

### 3.1. Recreating the Conda Environment

To ensure complete reproducibility of our results, please follow these steps to recreate the exact execution environment:

1.    Install Conda:   Ensure you have Anaconda or Miniconda installed on your system.
2.    Navigate to Repository:   Open your terminal or Anaconda Prompt and navigate to the root directory of this repository:
    ```bash
    cd /path/to/HPO-SDSS-Matching-Code
    ```
3.    Create and Activate Environment:   Use the provided `environment.yml` file to create the environment (named `xhc_v2.5`):
    ```bash
    conda env create -f environment.yml
    conda activate xhc_v2.5
    ```

### 3.2. Primary Software Used & Citation Disclosure

The analysis utilizes several key astronomical and machine learning libraries, including adapted code from the Swin Transformer architecture.

| Software/Library | Version (from environment.yml) | Key Function in Project |
| :--- | :--- | :--- |
|   PyTorch / Swin Transformer   | $\mathbf{2.5.1}$ | Deep Learning Model Training (Adapted from Liu et al. ${}^{17}$) |
|   Skyfield / sgp4   | $\mathbf{1.53}$ / $\mathbf{2.24}$ | TLE parsing, satellite state vectors, and orbital prediction |
|   Astropy   | $\mathbf{6.1.7}$ | Coordinate transformations (WCS) and FITS file handling |
|   OpenCV (cv2)   | $\mathbf{4.11.0}$ | Image preprocessing and traditional feature detection |

## 4. Running the Analysis (Mirroring Manuscript Sections)

The analysis is structured into three main directories that mirror the phases described in the Methods section (§2):

### 4.1. Phase 1: Deep Learning Classification (Directory: `1_model_train`)

This directory contains the Python scripts used to implement the   two-layer Swin Transformer filtering process   for streak identification.   Note:   These scripts include modifications to the original Swin Transformer module, as detailed in the paper.

| Script Name | Function |
| :--- | :--- |
| `model_use_swin_transformer_first.py` | Implements the   initial model training   and coarse filtering step. |
| `model_use_swin_transformer_second.py` | Implements the   secondary model training   using manually curated error samples. |

### 4.2. Phase 2: Orbital Matching (Directory: `2_object_match`)

This directory contains the core algorithm for cross-correlating the classified streaks with TLE data, covering coarse filtering, time-point sampling, and optimal matching (Methods §2.2).

| Script Name | Function |
| :--- | :--- |
| `object_match.py` |   Core matching algorithm  , calculating celestial coordinates (Skyfield) and performing angular difference filtering against TLE data. |

### 4.3. Phase 3: Final Data and Statistics (Manual Analysis)

The generated output data is stored in the `3_finally_match_result__include_filter_with_sdss_streak_information` directory.   The final statistical analysis, quantification, and plotting (Figures 3, 4, and Table 1) were performed manually using this derived catalog data.   The results are detailed directly within the manuscript's Results section (§3).

### Running Example Script

The data processing pipeline is complete upon running `object_match.py` (Phase 2). All subsequent statistical quantification is performed manually on the output CSV files found in the `3_finally_match_result...` directory.

## 5. Contact and Licensing
*   Contact:     Xiao Hengchu   (  xiaohengchu@Ynao.ac.cn  )
*   License:   This repository is licensed under the   MIT License  . The full terms and conditions are detailed in the `LICENSE` file at the root of the repository.

