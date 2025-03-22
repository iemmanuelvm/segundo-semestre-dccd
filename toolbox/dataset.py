import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CustomEEGDataset(Dataset):
    def __init__(self, eeg, eeg_eog, eeg_emg, window_size=512):
        self.signals = [(self.normalize(eeg), 0), 
                        (self.normalize(eeg_eog), 1), 
                        (self.normalize(eeg_emg), 2)]
        self.window_size = window_size
        self.samples = []
        
        for signal, label in self.signals:
            num_samples = len(signal) // window_size
            for i in range(num_samples):
                start = i * window_size
                end = start + window_size
                self.samples.append((torch.tensor(signal[start:end], dtype=torch.float32), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        return sample.unsqueeze(0), torch.tensor(label, dtype=torch.long)
    
    @staticmethod
    def normalize(signal):
        return (signal - np.mean(signal)) / np.std(signal)

def get_dataloaders(eeg, eeg_eog, eeg_emg, batch_size=32, shuffle=True):
    dataset = CustomEEGDataset(eeg, eeg_eog, eeg_emg)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
