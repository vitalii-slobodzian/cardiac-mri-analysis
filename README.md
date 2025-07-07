# Explainable Deep Learning for Cardiac MRI: Multi-Stage Segmentation, Cascade Classification, and Visual Interpretation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

A deep learning framework for automated cardiac MRI analysis, implementing multi-stage segmentation, cascade classification, and visual interpretation techniques. This repository supports the research presented in our preprint:  
**"Explainable Deep Learning for Cardiac MRI: Multi-Stage Segmentation, Cascade Classification, and Visual Interpretation"**  
[Read the preprint](https://www.preprints.org/user/home/submissions/details?hashkey=7a981606e841beeac104585f41bee45c&status=online)

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
  - [Custom Dataset Structure](#custom-dataset-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

This project addresses the challenge of automated cardiac structure analysis in MRI images through a novel multi-stage deep learning approach. Our pipeline combines precise segmentation of cardiac structures (RV, LV, and myocardium) with interpretable results, enabling both accurate medical image analysis and explainable AI techniques for clinical applications.

### Key Contributions

- Multi-stage segmentation pipeline for cardiac structures
- Cascade classification system for cardiac abnormality detection
- Visual interpretation methods for clinical explainability
- Comprehensive evaluation on the ACDC dataset

---

## Project Architecture

The pipeline consists of three main components:

1. **Segmentation Stage**
   - Preprocessing of NIFTI format medical images
   - Multi-class segmentation (RV, LV, myocardium)
   - ResNet-based U-Net architecture

2. **Classification Stage**
   - Feature extraction from segmented regions
   - Cascade classification for abnormality detection

3. **Interpretation Stage**
   - Saliency map generation
   - Region-based feature importance visualization

---

## Features

- NIFTI file processing for medical imaging
- Configurable data preprocessing pipeline (cropping, resizing, localization, class remapping)
- Training pipeline for multiple cardiac structures (RV, LV, Myo)
- Model evaluation using Dice metric
- Interpretability tools for clinical validation

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vitalii-slobodzian/cardiac-mri-analysis.git
   cd cardiac-mri-analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Data Preprocessing
```bash
python dataset_builder.py --config configs/preprocessing.yaml
```

### Model Training
```bash
python training.py --config configs/model.yaml
```

For detailed usage examples, see the notebooks in the `examples/` directory.

---

## Datasets

This project uses the ACDC (Automated Cardiac Diagnosis Challenge) dataset. To access the dataset:

1. Register at the [ACDC challenge website](https://acdc.creatis.insa-lyon.fr/)
2. Download and place the data in the `data/` directory
3. Follow the preprocessing steps in `dataset_builder.ipynb`

### Custom Dataset Structure

After preprocessing, the custom dataset is organized as follows:

```
custom-dataset/
├── cropped/
│   ├── training/
│   │   ├── rv/
│   │   │   ├── images/   # Cropped images for RV
│   │   │   └── masks/    # Corresponding masks for RV
│   │   ├── lv/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── myo/
│   │       ├── images/
│   │       └── masks/
│   └── testing/
│       ├── rv/
│       │   ├── images/
│       │   └── masks/
│       ├── lv/
│       │   ├── images/
│       │   └── masks/
│       └── myo/
│           ├── images/
│           └── masks/
├── localized/
│   ├── training/
│   │   ├── rv/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── lv/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── myo/
│   │       ├── images/
│   │       └── masks/
│   └── testing/
│       ├── rv/
│       │   ├── images/
│       │   └── masks/
│       ├── lv/
│       │   ├── images/
│       │   └── masks/
│       └── myo/
│           ├── images/
│           └── masks/
```

- Each `images/` folder contains `.png` files named as `patientXXX_frameYY_Z.png`.
- Each `masks/` folder contains corresponding mask images with the same naming convention.
- The structure is mirrored for both `cropped` and `localized` datasets, and for both `training` and `testing` splits.

---

## Reproducibility

- All experiments can be reproduced using the provided configuration files in `configs/`.
- Model checkpoints and detailed results will be made available upon publication.
- For exact hyperparameters and experiment settings, refer to the configuration YAML files.

---

## Citation

If you use this code in your research, please cite our preprint:

```bibtex
@article{slobodzian2024explainable,
    title={Explainable Deep Learning for Cardiac MRI: Multi-Stage Segmentation, 
           Cascade Classification, and Visual Interpretation},
    author={Slobodzian, V.; Barmak, O.; Radiuk, P.; Klymenko, L.; Krak, I.},
    journal={Preprints},
    year={2024},
    doi={10.20944/preprints202501.1280.v1},
    url={https://www.preprints.org/manuscript/202402.1234/v1}
}
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on submitting issues and pull requests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- ACDC Challenge organizers for the dataset
- Khmelnytskyi National University for computational resources

---

## Contact

- **Principal Investigator**: [Vitalii Slobodzian](mailto:vitalii.slobodzian@gmail.com)
- **GitHub Issues**: For technical questions and support

---

*Note: This repository is actively maintained. For the latest updates, please check our [releases page](https://github.com/vitalii-slobodzian/cardiac-mri-analysis/releases).*
