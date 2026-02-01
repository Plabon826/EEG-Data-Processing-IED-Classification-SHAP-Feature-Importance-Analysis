# EEG-Data-Processing-IED-Classification-SHAP-Feature-Importance-Analysis
This repository provides an EEG data processing pipeline for IED classification. It includes preprocessing, feature extraction, and training multiple classifiers (e.g., CatBoost, XGBoost, SVM). SHAP analysis is used to interpret feature and channel importance, enabling transparent and reproducible EEG analysis.

Overview

This repository provides code for processing EEG data, performing feature extraction, training classifiers to detect Intracranial Epileptiform Discharges (IED), and analyzing feature importance using SHAP. It is designed to clean EEG data, extract relevant features, and then use various machine learning classifiers to classify the data. SHAP is used to interpret model predictions and understand feature importance.

Repository Structure

Data Processing and Feature Extraction
	•	preprocessing.py: Functions for EEG data cleaning, preprocessing (e.g., ICA, filtering).
	•	feature_extraction.py: Functions for extracting time-domain, frequency-domain, and non-linear features from EEG data.
	•	quality_control.py: Functions to identify bad channels and reject corrupted epochs in EEG data.
	•	main.py: Main pipeline script that processes the EEG data, applies preprocessing, extracts features, and saves the results to CSV files.

Classifiers
	•	catboost_classifier.py: CatBoost classifier implementation for IED detection.
	•	decision_tree.py: Decision Tree classifier for IED detection.
	•	extra_trees.py: Extra Trees classifier for IED detection.
	•	lightgbm_classifier.py: LightGBM classifier for IED detection.
	•	logistic_regression.py: Logistic Regression classifier for IED detection.
	•	svm_classifier.py: Support Vector Machine (SVM) classifier for IED detection.
	•	xgboost_classifier.py: XGBoost classifier for IED detection.

SHAP Feature Importance Analysis
	•	Channel Wise Features Importance across Classifiers utilizing SHAP.py: Computes SHAP feature importance for each channel.
	•	Channel Wise Importance across Classifiers utilizing SHAP.py: Aggregates and visualizes SHAP values for each EEG channel.
	•	Features Importance across Classifiers utilizing SHAP.py: Aggregates SHAP importance across all features for each classifier.

Other Files
	•	requirements.txt: List of required Python dependencies (e.g., numpy, scipy, pandas, catboost, xgboost, lightgbm, shap).
	•	README.md: Documentation for the repository.


SHAP Analysis Process
	•	Top 10 Features: Displays top 10 features by SHAP importance.
	•	Channel-wise Importance: Aggregates SHAP values by EEG channel lead (i.e., channel-wise importance).
	•	Aggregated Feature Importance: Sums SHAP importance across all leads for each feature and visualizes the total feature importance.

Dataset

The dataset used in this repository is publicly available. You can access and download the dataset from the following link:

Dataset Link: https://doi.org/10.6084/m9.figshare.28069568.v2

Please ensure proper attribution and citation as outlined in the Data Citation section below.

Conclusion:

This repository provides an end-to-end workflow for EEG data processing, feature extraction, machine learning classifiers for IED detection, and SHAP analysis for interpreting classifier decisions. The goal is to identify which features and EEG channels are most influential in detecting IEDs, providing deeper insights into the data and model decisions.
