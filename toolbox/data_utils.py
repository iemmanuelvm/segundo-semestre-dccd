import os
import math
import numpy as np
import mne
import warnings

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

EEG_PATH = 'data'
EOG_ALL_EPOCHS = 'EOG_all_epochs.npy'
EMG_ALL_EPOCHS = 'EMG_all_epochs.npy'
EEG_ALL_EPOCHS = 'EEG_all_epochs.npy'

def split_signal(signal, parts=5):
    """
    Splits a 1D signal into 'parts' equal parts.
    Returns an array of shape (parts, segment_length).
    """
    total_length = len(signal)
    segment_length = total_length // parts
    segments = []
    for i in range(parts):
        start = i * segment_length
        end = start + segment_length
        segments.append(signal[start:end])
    return np.array(segments)

def show_data_information(signal, signal_type):
    print(f"Data type for {signal_type}:", type(signal))
    print(f"Data shape for {signal_type}:", signal.shape)

def load_signals():
    eog_data = np.load(os.path.join(EEG_PATH, EOG_ALL_EPOCHS))
    emg_data = np.load(os.path.join(EEG_PATH, EMG_ALL_EPOCHS))
    eeg_data = np.load(os.path.join(EEG_PATH, EEG_ALL_EPOCHS))
    
    show_data_information(eog_data, 'EOG')
    show_data_information(emg_data, 'EMG')
    show_data_information(eeg_data, 'EEG')
    
    return eeg_data, eog_data, emg_data

def generate_data(data, sfreq, ch_names, ch_types, scaling_dict):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data.reshape(1, -1), info)
    raw.plot(scalings=scaling_dict, title='Signal', show=True, block=True)

def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

def random_signal(signal, combine_num):
    random_result = []
    for i in range(combine_num):
        random_order = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_order, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        random_result.append(shuffled_dataset)
    random_result = np.array(random_result)
    return random_result

def add_noise(EEG_all_random, eog_data, emg_data):
    num_eeg_samples = EEG_all_random.shape[0]
    num_eog_samples = eog_data.shape[0]
    num_emg_samples = emg_data.shape[0]

    SNR_dB_eog = np.random.uniform(-7, 2, (num_eeg_samples))
    SNR_dB_emg = np.random.uniform(-7, 2, (num_eeg_samples))
    SNR_eog = 10 ** (0.1 * SNR_dB_eog)
    SNR_emg = 10 ** (0.1 * SNR_dB_emg)

    needed_repetitions_eog = int(np.ceil(num_eeg_samples / num_eog_samples))
    needed_repetitions_emg = int(np.ceil(num_eeg_samples / num_emg_samples))

    NOISE_all_random_eog = random_signal(signal=eog_data, combine_num=needed_repetitions_eog)
    NOISE_all_random_emg = random_signal(signal=emg_data, combine_num=needed_repetitions_emg)

    NOISE_all_random_eog = NOISE_all_random_eog.reshape(-1, eog_data.shape[1])
    NOISE_all_random_emg = NOISE_all_random_emg.reshape(-1, emg_data.shape[1])

    if NOISE_all_random_eog.shape[0] < num_eeg_samples:
        extra_needed = num_eeg_samples - NOISE_all_random_eog.shape[0]
        extra_noise = random_signal(signal=eog_data, combine_num=1).reshape(-1, eog_data.shape[1])
        NOISE_all_random_eog = np.concatenate((NOISE_all_random_eog, extra_noise), axis=0)

    if NOISE_all_random_emg.shape[0] < num_eeg_samples:
        extra_needed = num_eeg_samples - NOISE_all_random_emg.shape[0]
        extra_noise = random_signal(signal=emg_data, combine_num=1).reshape(-1, emg_data.shape[1])
        NOISE_all_random_emg = np.concatenate((NOISE_all_random_emg, extra_noise), axis=0)

    NOISE_all_random_eog = NOISE_all_random_eog[:num_eeg_samples]
    NOISE_all_random_emg = NOISE_all_random_emg[:num_eeg_samples]

    noiseEEG_EOG = []
    for i in range(num_eeg_samples):
        eeg = EEG_all_random[i]
        noise_eog = NOISE_all_random_eog[i]
        coef_eog = get_rms(eeg) / (get_rms(noise_eog) * SNR_eog[i])
        noise_eog = noise_eog * coef_eog
        noise_eeg_eog = eeg + noise_eog
        noiseEEG_EOG.append(noise_eeg_eog)

    noiseEEG_EMG = []
    for i in range(num_eeg_samples):
        eeg = EEG_all_random[i]
        noise_emg = NOISE_all_random_emg[i]
        coef_emg = get_rms(eeg) / (get_rms(noise_emg) * SNR_emg[i])
        noise_emg = noise_emg * coef_emg
        noise_eeg_emg = eeg + noise_emg
        noiseEEG_EMG.append(noise_eeg_emg)

    noiseEEG_EOG = np.array(noiseEEG_EOG)
    noiseEEG_EMG = np.array(noiseEEG_EMG)

    EEG_standardized = EEG_all_random / np.std(EEG_all_random)
    noiseEEG_EOG_standardized = noiseEEG_EOG / np.std(noiseEEG_EOG)
    noiseEEG_EMG_standardized = noiseEEG_EMG / np.std(noiseEEG_EMG)

    EEG = EEG_standardized.flatten()
    EEG_EOG = noiseEEG_EOG_standardized.flatten()
    EEG_EMG = noiseEEG_EMG_standardized.flatten()

    return EEG, EEG_EOG, EEG_EMG
