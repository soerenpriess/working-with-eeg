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

    # Wavelet-Decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # calc threshold
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    
    # Thresholding
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    #  Rekonstruktion
    return pywt.waverec(coeffs, wavelet)


def hjorth_parameters(signal):
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / activity)
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
    
    return activity, mobility, complexity

def spectral_entropy(signal, sf, method='welch', nperseg=None, normalize=False):
    if method == 'fft':
        freqs, psd = welch(signal, sf, nperseg=nperseg)
    elif method == 'welch':
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
    
    # normalize coefficients
    coeffs_normalized = coeffs / np.sum(coeffs)
    # calculate entropy
    return -np.sum(coeffs_normalized * np.log(coeffs_normalized + 1e-10))  # Hinzufügen von epsilon, um log(0) zu vermeiden

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

def left_tree(preprocessed_data):
    # alpha beta gammer 
    print("get alpha, beta, gamma...")
    start_time = time.time()
    freq_ranges = {
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 79)
    }
    filtered_data = {}
    # use fir filter
    for band, (low_freq, high_freq) in freq_ranges.items():
        filtered_data[band] = preprocessed_data.copy().filter(l_freq=low_freq, h_freq=high_freq, method='fir')
    end_time = time.time()
    print(f"Getting alpha, beta, gamma took {end_time - start_time:.4f} seconds\n")

    print("calc hjorth parameters and spectral entropy...")
    start_time = time.time()
    features = {}
    for band, data in filtered_data.items():
        features[band] = {
            'hjorth': [],
            'spectral_entropy': []
        }
        for ch in range(data.get_data().shape[0]):
            channel_data = data.get_data()[ch]
            h1, h2, h3 = hjorth_parameters(channel_data)
            se = spectral_entropy(channel_data, data.info['sfreq'])
            features[band]['hjorth'].append((h1, h2, h3))
            features[band]['spectral_entropy'].append(se)
    end_time = time.time()
    print(f"Calculating hjorth parameters and spectral entropy took {end_time - start_time:.4f} seconds\n")
    return features

def middle_tree(preprocessed_data):
    print("wavelet decomposition...")
    start_time = time.time()
    wavelet = 'db4'  # Daubechies Wavelet
    levels = 3       # amount of decomposition levels (D1, D2, D3)
    wavelet_coeffs = {}
    # wavelet decomposition for each channel
    for ch in range(preprocessed_data.get_data().shape[0]):
        channel_data = preprocessed_data.get_data(picks=[ch]).flatten()
        
        # Wavelet-Decomposition 
        coeffs = pywt.wavedec(channel_data, wavelet, level=levels)
        
        # Koeffizienten (D1, D2, D3)
        wavelet_coeffs[f'Channel_{ch+1}'] = {
            'D1': coeffs[1],
            'D2': coeffs[2],
            'D3': coeffs[3]
        }
    end_time = time.time()
    print(f"Wavelet decomposition took {end_time - start_time:.4f} seconds\n")

    print('calc wavelet energy and entrioy (from each D)...')
    start_time = time.time()
    wavelet_features = {}
    # calc energy and entropy for each channel and decomposition level
    for ch, coeffs in wavelet_coeffs.items():
        wavelet_features[ch] = {
            'D1': {
                'Energy': calculate_wavelet_energy(coeffs['D1']),
                'Entropy': calculate_entropy(coeffs['D1'])
            },
            'D2': {
                'Energy': calculate_wavelet_energy(coeffs['D2']),
                'Entropy': calculate_entropy(coeffs['D2'])
            },
            'D3': {
                'Energy': calculate_wavelet_energy(coeffs['D3']),
                'Entropy': calculate_entropy(coeffs['D3'])
            }
        }
    end_time = time.time()
    print(f"Calculating wavelet energy and entropy took {end_time - start_time:.4f} seconds\n")

    return wavelet_features


def right_tree(preprocessed_data):
    print("emd decomposition...")
    start_time = time.time()
    imfs_dict = {}
    for ch in preprocessed_data.ch_names:
        eeg_data = preprocessed_data.get_data(picks=[ch]).flatten()
        imfs = calculate_imfs(eeg_data)
        imfs_dict[ch] = imfs
    end_time = time.time()
    print(f"EMD decomposition took {end_time - start_time:.4f} seconds\n")

    print("imf energy and entropy... (from each IMF)")
    start_time = time.time()
    imf_features = {}
    for ch in imfs_dict:
        imf_features[ch] = []
        for imf in imfs_dict[ch]:
            energy, entropy_value = calculate_imf_features(imf)
            imf_features[ch].append((energy, entropy_value))
    end_time = time.time()
    print(f"Calculating IMF energy and entropy took {end_time - start_time:.4f} seconds\n")

    return imf_features


def calculate_differential_asymmetry(results):
    asymmetry = {}
    channel_pairs = [('F7', 'F8'), ('F3', 'F4'), ('T7', 'T8'), ('P7', 'P8'), ('O1', 'O2')]
    
    for left, right in channel_pairs:
        asymmetry[f"{left}-{right}"] = {
            'alpha': {
                'hjorth': [
                    results[left]['alpha']['hjorth'][i] - results[right]['alpha']['hjorth'][i]
                    for i in range(3)
                ],
                'spectral_entropy': results[left]['alpha']['spectral_entropy'] - results[right]['alpha']['spectral_entropy']
            },
            'wavelet': {
                f"D{i}": {
                    'Energy': results[left]['wavelet'][f"D{i}"]['Energy'] - results[right]['wavelet'][f"D{i}"]['Energy'],
                    'Entropy': results[left]['wavelet'][f"D{i}"]['Entropy'] - results[right]['wavelet'][f"D{i}"]['Entropy']
                } for i in range(1, 4)
            },
            'imf': {
                f"imf{i}": {
                    'energy': results[left][f"imf{i}"]['energy'] - results[right][f"imf{i}"]['energy'],
                    'entropy': results[left][f"imf{i}"]['entropy'] - results[right][f"imf{i}"]['entropy']
                } for i in range(1, 3) # TODO: für IMF3 implementieren (range 1, 4)
            }
        }
    
    return asymmetry




















def main():
    # Setze das Logging-Level von MNE auf WARNING
    mne.set_log_level('WARNING')

    overall_start_time = time.time()

    # Load data
    print("Loading and formatting raw data...")
    start_time = time.time()
    file_path = "../sampleData/S001R04.edf"
    desired_channels = ["F7", "F3", "T7", "P7", "O1", "O2", "P8", "T8", "F4", "F8"]
    raw = load_and_format_raw_data(file_path, desired_channels)
    end_time = time.time()
    print(f"Data loading and formatting took {end_time - start_time:.4f} seconds\n")

    # Preprocess data
    print("-----------------Preprocessing data-----------------")
    start_time = time.time()
    preprocessed_data = preprocess_data(raw)
    end_time = time.time()
    print(f"Data preprocessing took {end_time - start_time:.4f} seconds")
    print("-----------------------------------------------------\n")

    print("-----------------left tree-----------------")
    start_time = time.time()
    hjort_and_entropy = left_tree(preprocessed_data)
    end_time = time.time()
    print(f"Left tree took {end_time - start_time:.4f} seconds")
    print("-----------------------------------------------------\n")

    print("-----------------middle tree-----------------")
    start_time = time.time()
    wavelet_energy_entropy = middle_tree(preprocessed_data)
    end_time = time.time()
    print(f"Middle tree took {end_time - start_time:.4f} seconds")
    print("-----------------------------------------------------\n")

    print("-----------------right tree-----------------")
    start_time = time.time()
    imf_energy_and_entropy = right_tree(preprocessed_data)
    end_time = time.time()
    print(f"Right tree took {end_time - start_time:.4f} seconds")
    print("-----------------------------------------------------\n")

    overall_end_time_base_features = time.time()
    print(f"Overall runtime base features: {overall_end_time_base_features - overall_start_time:.4f} seconds")




    # print(json.dumps(hjort_and_entropy, indent=4))
    # print(json.dumps(wavelet_energy_entropy, indent=4))
    # print(json.dumps(imf_energy_and_entropy, indent=4))


    results = {}
    for i, ch in enumerate(preprocessed_data.ch_names):
        results[ch] = {
            'alpha': {
                'hjorth': hjort_and_entropy['alpha']['hjorth'][i],
                'spectral_entropy': hjort_and_entropy['alpha']['spectral_entropy'][i]
            },
            'beta': {
                'hjorth': hjort_and_entropy['beta']['hjorth'][i],
                'spectral_entropy': hjort_and_entropy['beta']['spectral_entropy'][i]
            },
            'gamma': {
                'hjorth': hjort_and_entropy['gamma']['hjorth'][i],
                'spectral_entropy': hjort_and_entropy['gamma']['spectral_entropy'][i]
            },
            'wavelet': wavelet_energy_entropy[f'Channel_{i+1}'],
            'imf1': {
                'energy': imf_energy_and_entropy[ch][0][0],
                'entropy': imf_energy_and_entropy[ch][0][1]
            },
            'imf2': {
                'energy': imf_energy_and_entropy[ch][1][0],
                'entropy': imf_energy_and_entropy[ch][1][1]
            },
            # 'imf3': {
            #     'energy': None, # imf_energy_and_entropy[ch][2][0],
            #     'entropy': None # imf_energy_and_entropy[ch][2][1]
            # } //TODO: für IMF3 implementieren
        }
    

    print("-----------------differential asymmetry-----------------")
    start_time = time.time()
    asymmetry = calculate_differential_asymmetry(results)
    end_time = time.time()
    print(f"Differential asymmetry took {end_time - start_time:.4f} seconds")
    

    results['asymmetry'] = asymmetry
    overall_end_time = time.time()
    print(f"Overall runtime: {overall_end_time - overall_start_time:.4f} seconds")

    # print(json.dumps(results, indent=4))





if __name__ == "__main__":
    main()