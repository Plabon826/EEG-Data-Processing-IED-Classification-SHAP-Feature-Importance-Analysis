IED Classifiers Repository

Overview:

This repository contains several classifiers for detecting Intracranial Epileptiform Discharges (IED) from EEG data. The classifiers include:
	•	CatBoost
	•	Decision Tree
	•	Extra Trees
	•	LightGBM
	•	Logistic Regression
	•	SVM (Support Vector Machine)
	•	XGBoost

Each classifier is trained using the SMOTE technique to balance the dataset and is evaluated using accuracy, classification report, and confusion matrix.

Files:
	•	catboost_classifier.py: CatBoost classifier implementation.
	•	decision_tree.py: Decision Tree classifier.
	•	extra_trees.py: Extra Trees classifier.
	•	lightgbm_classifier.py: LightGBM classifier.
	•	logistic_regression.py: Logistic Regression classifier.
	•	svm_classifier.py: SVM classifier.
	•	xgboost_classifier.py: XGBoost classifier.

Usage:
	1.	Install dependencies:
pip install -r requirements.txt

