# main.py
import os
import numpy as np
import pandas as pd
from preprocessing import preprocess_data
from feature_extraction import extract_features
from quality_control import check_epoch_quality, detect_bad_channels

def process_file(filepath, fs=500):
    """
    Process a single EEG data file, extract features and perform quality control.
    """
    filename = os.path.basename(filepath)
    data = np.load(filepath)

    # Preprocess the data
    data = preprocess_data(data, fs=fs)

    # Initialize result containers
    features = []
    qc_results = []

    # Loop through epochs
    epoch_samples = fs * 4  # Assume 4-second epochs
    n_epochs = data.shape[1] // epoch_samples

    for ep in range(n_epochs):
        start = ep * epoch_samples
        stop = (ep + 1) * epoch_samples

        epoch = data[:, start:stop]

        # Check for epoch quality
        reject = check_epoch_quality(epoch, fs)
        if reject:
            continue

        # Extract features
        epoch_features = []
        for lead in range(epoch.shape[0]):
            lead_features = extract_features(epoch[lead], fs)
            epoch_features.extend(lead_features)

        features.append(epoch_features)

        # Store QC results
        qc_results.append({"File": filename, "Epoch": ep + 1, "Rejected": int(reject)})

    return features, qc_results

def save_results(features, qc_results, output_file_features, output_file_qc):
    """
    Save extracted features and QC results to CSV files.
    """
    df_features = pd.DataFrame(features)
    df_qc = pd.DataFrame(qc_results)

    df_features.to_csv(output_file_features, index=False)
    df_qc.to_csv(output_file_qc, index=False)

if __name__ == "__main__":
    input_dir = "/path/to/your/data"
    output_file_features = "eeg_features.csv"
    output_file_qc = "epoch_qc.csv"

    file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')]

    all_features = []
    all_qc_results = []

    for filepath in file_list:
        features, qc_results = process_file(filepath)
        all_features.extend(features)
        all_qc_results.extend(qc_results)

    save_results(all_features, all_qc_results, output_file_features, output_file_qc)