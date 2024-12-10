# NSD-COCO Alignment and Reconstruction

This repository facilitates working with the NSD dataset and COCO annotations for alignment and image reconstruction tasks. Below are instructions for setting up the project, data preparation, and running the scripts.

---

## Contents
- [Requirements](#requirements)
- [Data Setup](#data-setup)
- [Helper Functions](#helper-functions)
- [Usage](#usage)
  - [Alignment](#alignment)
  - [Reconstruction](#reconstruction)
- [References](#references)

---

## Requirements
Ensure you have the following before getting started:

- Python 3.x
- Required libraries: see `requirements.txt` or install as needed.
- Access to the NSD dataset.
- COCO dataset annotations (indexed to align with NSD indexing).

---

## Data Setup

1. **Download Pre-Processed Data**:
   Pre-processed data for this project can be downloaded from the [Algonauts Challenges Dataset](http://algonauts.csail.mit.edu).

2. **Set Up the Data**:
   - Ensure the NSD dataset and COCO annotations are accessible.
   - Use the helper functions provided in the `NSD_access` folder to index and map between the datasets.
   - Instantiate the NSD data object with access to the original NSD dataset.

---

## Helper Functions

The `NSD_access` folder contains helper functions to retrieve COCO category annotations. These functions require:

- An instance of the NSD data object.
- Access to the original NSD dataset.

These helpers are used for indexing NSD images to corresponding COCO categories, enabling proper alignment and reconstruction tasks.

---

## Usage

### Alignment

To train the alignment model, run the `run_Alignment.py` script:

```bash
python run_Alignment.py
```

This script aligns image and text embeddings based on their shared semantic space.

### Reconstruction

To train the reconstruction model, run the `run_Reconstruction.py` script:

```bash
python run_Reconstruction.py
```

This script reconstructs images based on the aligned embeddings.

---

## References

- [Algonauts Challenge Dataset](http://algonauts.csail.mit.edu)
- NSD dataset documentation for accessing and interpreting NSD images.
- COCO dataset documentation for annotations and categories.

For any issues or questions, please refer to the dataset documentation or raise an issue in this repository.
