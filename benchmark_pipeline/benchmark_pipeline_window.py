import numpy as np
from scipy import signal
import pywt
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import entropy
from PyEMD import EMD
import time
import json

def load_and_format_raw_data(file_path, desired_channels):
    # load the edf file with mne
    raw = mne.io.read_raw_edf(file_path, preload=True)

    print(raw.ch_names)

    # rename channels without "." (dot) in the name
    new_ch_names = {ch_name: ch_name.replace('.', '') for ch_name in raw.info['ch_names']}
    raw.rename_channels(new_ch_names)

    # pick desired channels
    raw.pick_channels(desired_channels)

    # define channel positions (10-20 system)
    montage = mne.channels.make_standard_montage('standard_1020')
    positions = {ch: montage.get_positions()['ch_pos'][ch] for ch in desired_channels}

    # create new montage with the channel positions
    new_montage = mne.channels.make_dig_montage(positions, coord_frame='head')

    # set montage
    raw.set_montage(new_montage)

    return raw


def wavelet_denoising(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)


def hjorth_parameters(signal):
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / activity)
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
    return activity, mobility, complexity

def spectral_entropy(signal, sf, method='welch', nperseg=None, normalize=False):
    freqs, psd = welch(signal, sf, nperseg=nperseg)
    psd_norm = psd / psd.sum()
    se = entropy(psd_norm)
    if normalize:
        se /= np.log2(psd_norm.size)
    return se

def calculate_wavelet_energy(coeffs):
    return np.sum(coeffs**2)

def calculate_entropy(coeffs):
    if np.sum(coeffs) == 0:
        return 0
    coeffs_normalized = coeffs / np.sum(coeffs)
    return -np.sum(coeffs_normalized * np.log(coeffs_normalized + 1e-10))

def calculate_imfs(signal, num_imfs=3):
    emd = EMD()
    imfs = emd.emd(signal)
    return imfs[:num_imfs]

def calculate_imf_features(imf):
    energy = np.sum(imf**2)
    f, Pxx = welch(imf, fs=1, nperseg=len(imf))
    Pxx_norm = Pxx / np.sum(Pxx)
    entropy_value = entropy(Pxx_norm)
    return energy, entropy_value













def preprocess_data(raw):
    # detrend signal
    print("Detrending signal...")
    start_time = time.time()
    raw.apply_function(signal.detrend, overwrite_data=True)
    end_time = time.time()
    print(f"Detrending took {end_time - start_time:.4f} seconds\n")

    #notch filter
    print("Notch filtering signal...")
    start_time = time.time()
    notched_raw = raw.copy()
    notched_raw.notch_filter(60)
    end_time = time.time()
    print(f"Notch filtering took {end_time - start_time:.4f} seconds\n")

    # wavelet thresholding
    print("Wavelet denoising signal...")
    start_time = time.time()
    wavelet_raw = notched_raw.copy()

    for ch in wavelet_raw.ch_names:
        eeg_data = wavelet_raw.get_data(picks=[ch]).flatten()
        denoised_data = wavelet_denoising(eeg_data)
        wavelet_raw._data[wavelet_raw.ch_names.index(ch)] = denoised_data
    end_time = time.time()
    print(f"Wavelet denoising took {end_time - start_time:.4f} seconds\n")

    return wavelet_raw

def left_tree(preprocessed_epoch, sfreq):
    freq_ranges = {
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 79)
    }
    filtered_data = {}
    for band, (low_freq, high_freq) in freq_ranges.items():
        filtered_data[band] = mne.filter.filter_data(preprocessed_epoch, sfreq, low_freq, high_freq)
    
    features = {}
    for band, data in filtered_data.items():
        features[band] = {
            'hjorth': [],
            'spectral_entropy': []
        }
        for ch in range(data.shape[0]):
            channel_data = data[ch]
            h1, h2, h3 = hjorth_parameters(channel_data)
            se = spectral_entropy(channel_data, sfreq)
            features[band]['hjorth'].append((h1, h2, h3))
            features[band]['spectral_entropy'].append(se)
    
    return features

def middle_tree(preprocessed_epoch):
    wavelet = 'db4'
    levels = 3
    wavelet_features = {}
    
    for ch in range(preprocessed_epoch.shape[0]):
        channel_data = preprocessed_epoch[ch]
        coeffs = pywt.wavedec(channel_data, wavelet, level=levels)
        wavelet_features[f'Channel_{ch+1}'] = {
            f'D{i}': {
                'Energy': calculate_wavelet_energy(coeffs[i]),
                'Entropy': calculate_entropy(coeffs[i])
            } for i in range(1, levels+1)
        }
    
    return wavelet_features


def right_tree(preprocessed_epoch):
    imf_features = {}
    for ch in range(preprocessed_epoch.shape[0]):
        channel_data = preprocessed_epoch[ch]
        imfs = calculate_imfs(channel_data)
        imf_features[f'Channel_{ch+1}'] = [
            calculate_imf_features(imf) for imf in imfs
        ]
    
    return imf_features






























def calculate_differential_asymmetry_features(preprocessed_epoch, sfreq, ch_names):
    print("Calculating features from differential asymmetry...")
    start_time = time.time()
    channel_pairs = [
        ('F7', 'F8'),
        ('F3', 'F4'),
        ('T7', 'T8'),
        ('P7', 'P8'),
        ('O1', 'O2')
    ]
    features = {}
    for ch1, ch2 in channel_pairs:
        left_data = preprocessed_epoch[ch_names.index(ch1)]
        right_data = preprocessed_epoch[ch_names.index(ch2)]
        asymmetry = left_data - right_data
        
        # Berechne Wavelet-Koeffizienten
        wavelet = 'db4'
        levels = 3
        coeffs = pywt.wavedec(asymmetry, wavelet, level=levels)
        
        features[f'{ch1}-{ch2}'] = {
            'Hjorth': hjorth_parameters(asymmetry),
            'SpectralEntropy': spectral_entropy(asymmetry, sfreq),
            'Wavelet': {
                f'D{i}': {
                    'Energy': calculate_wavelet_energy(coeffs[i]),
                    'Entropy': calculate_entropy(coeffs[i])
                } for i in range(1, levels+1)
            },
            'IMF': []
        }
        
        # Berechne IMF Features
        imfs = calculate_imfs(asymmetry)
        for i, imf in enumerate(imfs, 1):
            energy, entropy_value = calculate_imf_features(imf)
            features[f'{ch1}-{ch2}']['IMF'].append({
                f'IMF{i}': {
                    'Energy': energy,
                    'Entropy': entropy_value
                }
            })

    end_time = time.time()
    print(f"Differential asymmetry features calculation took {end_time - start_time:.4f} seconds")
    return features

def preprocess_epoch(epoch, sfreq):
    # Detrending
    epoch = signal.detrend(epoch)
    
    # Notch-Filter
    b, a = signal.iirnotch(60, 30, sfreq)
    epoch = signal.filtfilt(b, a, epoch)
    
    # Wavelet-Denoising
    for ch in range(epoch.shape[0]):
        epoch[ch] = wavelet_denoising(epoch[ch])
    
    return epoch



def main():
    overall_start_time = time.time()

    # Setze das Logging-Level von MNE auf WARNING
    mne.set_log_level('WARNING')

    # Daten laden
    print("Lade und formatiere Rohdaten...")
    file_path = "../sampleData/S001R04.edf"
    desired_channels = ["F7", "F3", "T7", "P7", "O1", "O2", "P8", "T8", "F4", "F8"]
    raw = load_and_format_raw_data(file_path, desired_channels)

    # Epochen erstellen
    epoch_duration = 4  # Sekunden
    overlap = 0.5  # 50% Überlappung
    data = raw.get_data()
    sfreq = raw.info['sfreq'] #sampling frequency in Hz (160Hz)
    epoch_samples = int(epoch_duration * sfreq)
    step = int(epoch_samples * (1 - overlap))

    total_preprocessing_time = 0
    total_feature_calculation_time = 0

    for start in range(0, data.shape[1] - epoch_samples + 1, step):
        epoch_start_time = time.time()
        
        end = start + epoch_samples
        epoch = data[:, start:end]

        print(f"\nVerarbeite Epoche von {start/sfreq:.4f}s bis {end/sfreq:.4f}s")

        # Preprocessing
        print("Preprocessing der Epoche...")
        preprocess_start = time.time()
        preprocessed_epoch = preprocess_epoch(epoch, sfreq)
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
        total_preprocessing_time += preprocess_time
        print(f"Preprocessing-Zeit: {preprocess_time:.4f} Sekunden")

        # Feature-Berechnung
        print("Berechne Features...")
        feature_start = time.time()

        print("Linker Baum...")
        hjort_and_entropy = left_tree(preprocessed_epoch, sfreq)

        print("Mittlerer Baum...")
        wavelet_energy_entropy = middle_tree(preprocessed_epoch)

        print("Rechter Baum...")
        imf_energy_and_entropy = right_tree(preprocessed_epoch)

        print("alpha diffential asymmetry...")
        asymmetry = calculate_differential_asymmetry_features(preprocessed_epoch, sfreq, raw.ch_names)

        feature_end = time.time()
        feature_time = feature_end - feature_start
        total_feature_calculation_time += feature_time
        print(f"Feature-Berechnungszeit: {feature_time:.4f} Sekunden")

        epoch_end_time = time.time()
        epoch_total_time = epoch_end_time - epoch_start_time
        print(f"Gesamtzeit für diese Epoche: {epoch_total_time:.4f} Sekunden")

    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time

    print(f"\nGesamtlaufzeit: {total_time:.4f} Sekunden")
    print(f"Gesamte Preprocessing-Zeit: {total_preprocessing_time:.4f} Sekunden")
    print(f"Gesamte Feature-Berechnungszeit: {total_feature_calculation_time:.4f} Sekunden")

if __name__ == "__main__":
    main()