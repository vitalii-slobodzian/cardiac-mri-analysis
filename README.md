# Explainable Deep Learning for Cardiac MRI Analysis

[![Paper](https://img.shields.io/badge/Preprint-10.20944/preprints202501.1280.v1-green)](https://doi.org/10.20944/preprints202501.1280.v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/vitalii-slobodzian/cardiac-mri-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/vitalii-slobodzian/cardiac-mri-analysis/actions/workflows/ci.yml)

This repository contains the official implementation for the paper **"Explainable Deep Learning for Cardiac MRI: Multi-Stage Segmentation, Cascade Classification, and Visual Interpretation"**. Our work introduces a complete framework for automated, accurate, and transparent analysis of cardiac MRI scans, bridging the gap between advanced deep learning models and clinical interpretability.

![Overall Method Chart](https://raw.githubusercontent.com/vitalii-slobodzian/cardiac-mri-analysis/refs/heads/develop/img/fig_1.png)
*(Caption: The proposed task decomposition for cardiac MRI processing, illustrating the process flow from input MRI scans through three sequential processing stages—segmentation, classification, and interpretation.)*

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Our Methodology](#our-methodology)
  - [1. Multi-Stage Segmentation](#1-multi-stage-segmentation)
  - [2. Cascade Classification](#2-cascade-classification)
  - [3. Visual Interpretation](#3-visual-interpretation)
- [Performance Highlights](#performance-highlights)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

Automated analysis of Cardiac MRI is critical for efficient and objective diagnosis of cardiovascular diseases. However, existing methods often struggle with segmentation accuracy, confusion between similar pathologies, and the "black box" nature of deep learning models. This project introduces a novel three-stage pipeline designed to overcome these challenges. We provide a robust solution that not only achieves state-of-the-art accuracy in segmenting cardiac structures and classifying diseases but also generates clinically relevant explanations for its decisions.

---

## Key Features

- **High-Precision Segmentation**: Implements a two-step localization and segmentation approach using U-Net with ResNet backbones to accurately delineate the Left Ventricle (LV), Right Ventricle (RV), and Myocardium (Myo).
- **Accurate Disease Classification**: A cascade of specialized binary classifiers effectively distinguishes between five cardiac conditions: Normal (NOR), Dilated Cardiomyopathy (DCM), Hypertrophic Cardiomyopathy (HCM), Myocardial Infarction (MINF), and Abnormal Right Ventricle (ARV).
- **Clinical Explainability (XAI)**: Translates model predictions into intuitive, metric-based visualizations, including 17-segment bull's-eye plots and key cardiac performance indicators (e.g., ejection fraction, ventricular volumes), making the AI's reasoning transparent to clinicians.
- **End-to-End Workflow**: Provides a complete pipeline from raw NIFTI data preprocessing to final diagnostic interpretation.
- **Reproducible Research**: Includes detailed configuration files, pre-processed datasets, and clear instructions to fully reproduce our findings.

---

## Our Methodology

Our framework decomposes the complex task of cardiac analysis into three manageable and specialized stages:

### 1. Multi-Stage Segmentation

Instead of a single model, we use a sequence of six deep learning models to first **localize** the cardiac structures and then perform fine-grained **segmentation**. This hierarchical approach allows the models to focus on relevant regions, significantly improving accuracy. In many cases, our model provided more precise segmentations than the original expert annotations, correctly identifying or excluding regions that were mislabeled.

| Expert-Provided Mask | Our Method's Generated Mask | Difference Map |
|:---:|:---:|:---:|
| ![Expert Mask](https://raw.githubusercontent.com/vitalii-slobodzian/cardiac-mri-analysis/develop/img/fig_2a.png) | ![Generated Mask](https://raw.githubusercontent.com/vitalii-slobodzian/cardiac-mri-analysis/develop/img/fig_2b.png) | ![Difference Map](https://raw.githubusercontent.com/vitalii-slobodzian/cardiac-mri-analysis/develop/img/fig_2c.png) |
| **(a)** | **(b)** | **(c)** |

*(Caption: A visual comparison showcasing the expert-provided mask (a), our method’s generated mask (b), and a difference map (c). This example highlights cases where our model corrected inaccuracies present in the original ground truth.)*

### 2. Cascade Classification

To overcome class confusion common in multi-class medical classification, we designed a cascade of binary classifiers. This model first separates healthy from pathological cases and then progressively narrows down the specific diagnosis, leading to higher overall accuracy.

<img src="https://raw.githubusercontent.com/vitalii-slobodzian/cardiac-mri-analysis/refs/heads/develop/img/fig_3.png" alt="Cascade Classification Model" width="600"/>

*(Caption: The structure of our cascade classification model, which breaks down a five-class problem into a series of simpler binary classification tasks.)*

### 3. Visual Interpretation

The final and most critical stage is making the results understandable. Our interpretation module takes the segmentation and classification outputs and calculates clinically established metrics. These are presented in a comprehensive dashboard, providing a clear, quantitative, and visual summary to support a clinician's final diagnosis.

<img src="https://raw.githubusercontent.com/vitalii-slobodzian/cardiac-mri-analysis/refs/heads/develop/img/fig_4.png" alt="DCM Interpretation" width="1200"/>

*(Caption: Example of the final interpretation output for a patient with Dilated Cardiomyopathy (DCM), showing key metrics and a 17-segment model of myocardial wall thickness.)*

---

## Performance Highlights

Our methods set a new standard for accuracy on the ACDC dataset.

**Segmentation Performance (Dice Coefficient):**

| Phase | Left Ventricle (LV) | Right Ventricle (RV) | Myocardium (Myo) |
|:---:|:---:|:---:|:---:|
| **End-Diastole (ED)** | **0.974** | **0.947** | 0.896 |
| **End-Systole (ES)** | **0.940** | **0.915** | **0.920** |

**Classification Performance:**
Our cascade classifier achieved an **overall accuracy of 97%**, significantly outperforming standard ResNet architectures which scored between 72-84% on the same task.

---

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/vitalii-slobodzian/cardiac-mri-analysis.git
    cd cardiac-mri-analysis
    ```

2. **Choose an environment option (Python 3.9+):**

   - Using Conda (recommended for scientific stacks):

     ```bash
     conda create -n cardio-mri python=3.9 -y
     conda activate cardio-mri
     # Core deps (current subset used by segmentation code)
     pip install -r segmentation/requirements.txt
     # Dev tools
     pip install black pytest jupyterlab
     ```

   - Using venv (standard Python):

     ```bash
     python -m venv .venv
     # Linux/macOS
     source .venv/bin/activate
     # Windows (PowerShell)
     .venv\Scripts\Activate.ps1
     # Install dependencies
     pip install -r segmentation/requirements.txt
     pip install black pytest jupyterlab
     ```

3. **Optional: verify GPU setup** (if using CUDA-enabled PyTorch). Ensure your CUDA/PyTorch versions are compatible; otherwise the project runs on CPU.

---

## Usage

The entire pipeline can be run using simple CLI commands with YAML configuration files.

1. **Preprocess the Data**:
    This script converts the raw ACDC NIFTI files into the required PNG format and folder structure.

    ```bash
    python dataset_builder.py --config configs/preprocessing.yaml
    ```

2. **Train the Models**:
    Train the segmentation and classification models using the specified configurations.

    ```bash
    python training.py --config configs/model.yaml
    ```

3. **Run Inference**:
    Perform segmentation and classification on a new patient's data.

    ```bash
    python inference.py --config configs/inference.yaml --patient_id <patient_id>
    ```

For more detailed examples and tutorials, please refer to the notebooks in the `examples/` directory.

---

## Development Guide

Follow these steps when developing locally.

1. Format code with Black:

   ```bash
   black .
   ```

2. Run tests with Pytest:

   ```bash
   pytest -q
   ```

   The repository includes example unit tests under `tests/` covering:
   - `DiceMetric.multi_dice` computation for perfect and partial overlap
   - `NiiReader.parse_config` parsing of typical ACDC metadata

3. Work with notebooks:

   ```bash
   jupyter lab
   ```

   Prefer lightweight, deterministic cells; avoid committing large outputs.

4. Continuous Integration (CI):

   - Every push/PR to `main` or `develop` runs GitHub Actions: `black --check .` and `pytest -q`.
   - Fix formatting locally with `black .` and ensure tests pass before opening a PR.
   - CI uses pip caching and pinned deps from `segmentation/requirements.txt` (e.g., `torch==2.5.1`, `torchvision==0.20.1`, `fastai==2.7.18`) for reproducibility and faster builds.

## Dataset

This study was conducted using the **Automated Cardiac Diagnosis Challenge (ACDC) dataset**.

- **Original Dataset**: To access the raw data, you must register at the [ACDC Challenge Website](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).
- **Pre-processed Dataset**: For convenience and full reproducibility, we have made our exact pre-processed dataset publicly available. This includes all training and testing splits used in the paper.
    - **[Download our Pre-processed Data (Google Drive)](https://drive.google.com/drive/folders/1qBpZR2LvrWwW70OLAJbOx74eWBlxtkA2?usp=sharing)**

After running the preprocessing script, your `data/` directory will be structured as follows:

```markdown
data/custom-dataset/
├── cropped/
│   ├── training/
│   │   ├── rv/, lv/, myo/
│   └── testing/
│       ├── rv/, lv/, myo/
└── localized/
    ├── training/
    │   ├── rv/, lv/, myo/
    └── testing/
        ├── rv/, lv/, myo/
```

---

## Reproducibility

We are committed to open and reproducible science.

- All model configurations, hyperparameters, and training settings are defined in the YAML files within the `configs/` directory.
- Trained model weights and checkpoints will be made available upon the final publication of the manuscript.

---

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{slobodzian2025explainable,
  title={Explainable Deep Learning for Cardiac MRI: Multi-Stage Segmentation, Cascade Classification, and Visual Interpretation},
  author={Slobodzian, Vitalii and Barmak, Oleksandr and Radiuk, Pavlo and Klymenko, Liliana and Krak, Iurii},
  journal={Preprints.org},
  year={2025},
  version={v1},
  doi={10.20944/preprints202501.1280.v1},
  url={https://doi.org/10.20944/preprints202501.1280.v1}
}
```

---

## Contributing

Contributions are welcome! If you'd like to improve the code, report a bug, or suggest a feature, please see our [Contributing Guidelines](CONTRIBUTING.md) and open an issue or pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

We gratefully acknowledge the organizers of the ACDC Challenge for providing the public dataset and Khmelnytskyi National University for providing the computational resources necessary for this research.

## Contact

- **Principal Investigator**: Vitalii Slobodzian ([vitalii.slobodzian@gmail.com](mailto:vitalii.slobodzian@gmail.com))
- For technical questions or bug reports, please open an issue on the [GitHub Issues page](https://github.com/vitalii-slobodzian/cardiac-mri-analysis/issues).
