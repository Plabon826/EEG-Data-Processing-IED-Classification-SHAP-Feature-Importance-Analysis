# EEG Data Processing and Feature Extraction

This repository contains code to preprocess EEG data, extract features, and perform quality control on EEG epochs. The code involves multiple stages such as filtering, Independent Component Analysis (ICA), and feature extraction.

## Repository Structure

- **`preprocessing.py`**: Functions for EEG data cleaning and preprocessing.
- **`feature_extraction.py`**: Functions for extracting time-domain, frequency-domain, and non-linear features.
- **`quality_control.py`**: Functions for identifying bad channels and rejecting corrupted epochs.
- **`main.py`**: Main file to run the pipeline and process the EEG data, saving the results to CSV.
- **`requirements.txt`**: List of Python dependencies (e.g., numpy, scipy, pandas).
- **`README.md`**: Documentation for the repository.

## Installation

Clone the repository and install dependencies

