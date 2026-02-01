# quality_control.py
import numpy as np
from scipy.signal import periodogram

# QC Functions for detecting bad channels and epochs
def detect_bad_channels(data, fs, f0, var_z_thresh=4.5, corr_min_thresh=0.20, line_noise_ratio_thresh=0.90, max_bad_fraction=0.40):
    """
    Detect bad channels based on standard deviation, correlation, and line noise.
    """
    X = np.nan_to_num(data, nan=0.0).copy()
    n_ch, _ = X.shape

    # Variance checks
    stds = np.std(X, axis=1)
    bad_flat = stds < 1e-8
    logvars = np.log10(np.var(X, axis=1) + 1e-12)
    med = np.median(logvars)
    mad = np.median(np.abs(logvars - med)) + 1e-12
    z = (logvars - med) / (1.4826 * mad)
    bad_var = np.abs(z) > var_z_thresh

    # Line-noise ratio check
    def _ln_ratio(x):
        freqs, Pxx = periodogram(x, fs=fs)
        band_all = (freqs >= 0.5) & (freqs <= 45)
        band_ln = (freqs >= (f0 - 1)) & (freqs <= (f0 + 1))
        p_all = np.trapz(Pxx[band_all], freqs[band_all]) + 1e-12
        p_ln = np.trapz(Pxx[band_ln], freqs[band_ln]) + 1e-12
        return p_ln / p_all

    ln_ratios = np.array([_ln_ratio(X[i]) for i in range(n_ch)])
    bad_ln = ln_ratios > line_noise_ratio_thresh

    bad = bad_flat | bad_var | bad_ln
    bad_idx = np.where(bad)[0].tolist()

    # Safety check: cap the percentage of bad channels
    cap = int(np.floor(max_bad_fraction * n_ch))
    if len(bad_idx) > cap:
        bad = bad_flat | bad_var | bad_corr
        bad_idx = np.where(bad)[0].tolist()

    return bad_idx

def check_epoch_quality(epoch, fs, ptp_thresh=6.0, high_freq_thresh=0.6):
    """
    Perform QC on each epoch. Reject if high frequency fraction or peak-to-peak is too high.
    """
    ptp_vals = np.ptp(epoch, axis=1)
    fail_ptp = np.any(ptp_vals > ptp_thresh)

    # High-frequency fraction
    p_all = np.array([bandpower(epoch[i], fs, (0.5, 40)) for i in range(epoch.shape[0])])
    p_hf = np.array([bandpower(epoch[i], fs, (20, 40)) for i in range(epoch.shape[0])])
    hf_frac = p_hf / (p_all + 1e-12)
    fail_hf = np.any(hf_frac > high_freq_thresh)

    reject = bool(fail_ptp or fail_hf)

    return reject