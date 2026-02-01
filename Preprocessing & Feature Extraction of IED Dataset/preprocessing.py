# preprocessing.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, detrend
import warnings

# =========================
# === FILTER DESIGNERS ====
# =========================
def design_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def design_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return b, a

def design_notch_filters(fs, base_hz, num_harmonics=0, Q=30):
    """
    Return a list of (b, a) for f0 and up to num_harmonics harmonics (< Nyquist).
    Example: num_harmonics=1 -> [f0, 2*f0]
    """
    if base_hz is None or base_hz <= 0:
        return []
    nyq = fs / 2.0
    filters = []
    k = 1
    while (k * base_hz) < nyq and k <= (1 + num_harmonics):
        f0 = k * base_hz
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        filters.append((b, a))
        k += 1
    return filters

def apply_filter_chain(x, ba_list, axis=-1):
    y = x
    for (b, a) in ba_list:
        y = filtfilt(b, a, y, axis=axis)
    return y

# =========================
# ========== UTILS ========
# =========================
def bandpower(x, fs, freq_band):
    from scipy.integrate import trapezoid
    freqs, Pxx = periodogram(x, fs=fs)
    mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    return trapezoid(Pxx[mask], freqs[mask])

def preprocess_data(data, fs=500, line_freq_hz=50, first_n_eeg=19):
    """
    Process EEG data by detrending, applying notch filters, and band-pass filtering.
    """
    # Choose line frequency
    if line_freq_hz == 'auto':
        # Automatically detect line frequency (50/60 Hz)
        pass  # Add your own automatic detection method if needed

    # Detrend signal
    data = detrend(data, axis=-1)

    # Apply notch filters
    notch_filters = design_notch_filters(fs, base_hz=line_freq_hz)
    data = apply_filter_chain(data, notch_filters, axis=-1)

    # Apply band-pass filter
    bp_b, bp_a = design_bandpass(0.5, 40, fs)
    data = filtfilt(bp_b, bp_a, data, axis=-1)
    
    return data