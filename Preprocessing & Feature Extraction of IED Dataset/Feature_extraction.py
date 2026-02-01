# feature_extraction.py
import numpy as np
from scipy.signal import periodogram
from scipy.stats import skew, kurtosis

# Feature extraction functions
def extract_time_domain_features(signal):
    mean = np.mean(signal)
    median = np.median(signal)
    std = np.std(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    ptp = np.ptp(signal)  # Peak-to-peak amplitude

    return [mean, median, std, skewness, kurt, ptp]

def extract_frequency_domain_features(signal, fs):
    power_delta = bandpower(signal, fs, (0.5, 4))
    power_theta = bandpower(signal, fs, (4, 8))
    power_alpha = bandpower(signal, fs, (8, 13))
    power_beta = bandpower(signal, fs, (13, 30))
    alpha_beta_ratio = power_alpha / max(power_beta, 1e-10)

    return [power_delta, power_theta, power_alpha, power_beta, alpha_beta_ratio]

def extract_non_linear_features(signal):
    # Sample entropy, Approximate entropy, and other non-linear features
    # These functions can be added for advanced feature extraction
    return []

def extract_features(signal, fs):
    # Extract time-domain features
    time_domain_features = extract_time_domain_features(signal)

    # Extract frequency-domain features
    frequency_domain_features = extract_frequency_domain_features(signal, fs)

    # Extract non-linear features (placeholder for now)
    non_linear_features = extract_non_linear_features(signal)

    return time_domain_features + frequency_domain_features + non_linear_features