from data_utils import load_signals, random_signal, add_noise
from model import load_model
from visualization import animate_random_segments
import numpy as np

def main():
    # Load original signals
    eeg_data, eog_data, emg_data = load_signals()
    
    # Get random EEG for subsequent noise addition
    EEG_all_random = random_signal(signal=eeg_data, combine_num=1).squeeze()
    
    # Generate signals with added noise: EEG, EEG+EOG, and EEG+EMG
    EEG, EEG_EOG, EEG_EMG = add_noise(EEG_all_random, eog_data, emg_data)
    
    # Verify that the signals have sufficient length
    needed_length = 512 * 5  # 5 segments of 512 samples
    if len(EEG) < needed_length or len(EEG_EOG) < needed_length or len(EEG_EMG) < needed_length:
        raise ValueError("One of the signals does not have the required minimum length.")
    
    # Optionally: select a portion of the signal for animation
    final_EEG     = EEG[:512*50]
    final_EEG_EOG = EEG_EOG[:512*50]
    final_EEG_EMG = EEG_EMG[:512*50]
    
    # Load the trained model
    model = load_model('time_domain_nn_model.pth')
    
    # Start the animation: in each iteration a random signal among the three is selected
    animate_random_segments(model, final_EEG, final_EEG_EOG, final_EEG_EMG, win_size=512, num_segments=5)

if __name__ == "__main__":
    main()
