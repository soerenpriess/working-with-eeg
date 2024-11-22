import mne
import numpy as np

file_path = "sampleData/S001R04.edf"

# Lade die EDF-Datei mit MNE
raw = mne.io.read_raw_edf(file_path, preload=True)

# Kanalnamen ohne "." (Punkt) in den Namen umbenennen
new_ch_names = {ch_name: ch_name.replace('.', '') for ch_name in raw.info['ch_names']}
raw.rename_channels(new_ch_names)

# Liste der gewünschten Kanäle definieren (10-20 System; gewünschte Kanäle sind die 14 emotiv Epoc Kanäle, alledings existieren in dem Datensatz nur 10 davon)
desired_channels = ["F7", "F3", "T7", "P7", "O1", "O2", "P8", "T8", "F4", "F8"]
raw.pick_channels(desired_channels)

# Definition der Kanalpositionen
montage = mne.channels.make_standard_montage('standard_1020')
positions = {ch: montage.get_positions()['ch_pos'][ch] for ch in desired_channels}

# Neue Montage mit den Kanalpositionen erstellen
new_montage = mne.channels.make_dig_montage(positions, coord_frame='head')

# Montage setzten
raw.set_montage(new_montage)

raw.plot(title="Originaldaten", duration=5, n_channels=len(desired_channels))

input("Press Enter to continue...")


# Detrending anwenden
raw.apply_function(lambda x: x - np.mean(x), picks=desired_channels)

# 50 Hz Notch-Filter anwenden
raw.notch_filter(freqs=[50], picks=desired_channels)

# Plotte das PSD-Diagramm nach Pre-Processing
raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
raw.plot(duration=5, n_channels=len(desired_channels), title="Pre-Processed Daten")

input("Press Enter to continue...")