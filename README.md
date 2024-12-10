# Reconstructing Mental Imagery with Semantically Aligned Embeddings

This repository facilitates working with the NSD dataset and COCO annotations for alignment and image reconstruction tasks. Below are instructions for setting up the project, data preparation, and running the scripts.

---

## Contents
- [Requirements](#requirements)
- [Data Setup](#data-setup)
- [Usage](#usage)
  - [Alignment](#alignment)
  - [Reconstruction](#reconstruction)
- [References](#references)

---

## Requirements
Ensure you have the following before getting started:

- Required libraries: see `requirements.txt` or install as needed.

---

## Data Setup

1. **Download Pre-Processed Data**:
   Pre-processed data for this project can be downloaded from the [Algonauts Challenges Dataset](http://algonauts.csail.mit.edu). 

2. **Set Up the Data**:
   - Download data from the Algonauts website. Put this data into the `NSD` folder.
   - In the `NSD` folder, you will find three CSV files: `categories.csv`, `largest_categories.csv`, and `largest_categories_weights.csv`, which contain the category annotation information of the 73K images used in the NSD dataset.
   - As described in the Algonauts challenge, each image contains an index that corresponds to the row in the 73K dataset it originated from.
   - These three CSV files were generated from helper functions in `NSD/NSD Access Helper Functions`, which required an instance of the original (not Algonauts) NSD data for indexing purposes.

---

## Usage

### Alignment

To train the alignment model, run the `run_alignment.py` script:

```bash
python3 run_alignment.py
```

This script aligns image and text embeddings based on their shared semantic space.

### Reconstruction

To train the reconstruction model, run the `run_reconstruction.py` script:

```bash
python3 run_reconstruction.py
```

This script reconstructs images based on the aligned embeddings.

---

## References

- [Algonauts Challenge Dataset](http://algonauts.csail.mit.edu)
- [Microsoft COCO Categories](https://cocodataset.org/#home)

For any issues or questions, please refer to the dataset documentation or raise an issue in this repository.
