import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class_names = {
    0: "EEG",
    1: "Chewing",
    2: "Electrode pop",
    3: "Eye movement",
    4: "Muscle",
    5: "Shiver",
}


class AlignedEEGDataset(Dataset):
    def __init__(self, signal, window_size=512, n_fft=128, hop_length=64, norm_type="zscore"):
        self.signal = signal.astype(np.float32)
        self.window_size = window_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.norm_type = norm_type
        self.segments = self.segment_signal()

    def segment_signal(self):
        segments = []
        for i in range(0, len(self.signal), self.window_size):
            segment = self.signal[i:i + self.window_size]
            if len(segment) == self.window_size:
                segments.append(segment)
        return segments

    def apply_stft(self, segment):
        eeg_tensor = torch.tensor(segment, dtype=torch.float32)
        stft_result = torch.stft(eeg_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                                 return_complex=True).abs()
        return stft_result

    def normalize_minmax(self, tensor):
        min_val = tensor.min(
            dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = tensor.max(
            dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    def normalize_zscore(self, tensor):
        mean = tensor.mean(dim=(-1, -2), keepdim=True)
        std = tensor.std(dim=(-1, -2), keepdim=True) + 1e-8
        return (tensor - mean) / std

    def normalize(self, tensor):
        if self.norm_type == "minmax":
            return self.normalize_minmax(tensor)
        elif self.norm_type == "zscore":
            return self.normalize_zscore(tensor)
        else:
            return tensor

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        x_feat_stft = self.apply_stft(segment)
        x_feat_stft = self.normalize(x_feat_stft)
        return x_feat_stft.unsqueeze(0)


def load_model():
    from torch import nn

    class ResidualBlock2D(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock2D, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
            )
            self.downsample = downsample
            self.relu = nn.ReLU()

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            return self.relu(out)

    class ResNet2D(nn.Module):
        def __init__(self, block, layers, num_classes=6):
            super(ResNet2D, self).__init__()
            self.in_channels = 64

            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
            self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

            self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, block, out_channels, blocks, stride=1):
            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels,
                              kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels),
                )
            layers = []
            layers.append(
                block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(block(self.in_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.global_avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.fc(x)
            return logits, x  # (logits, features)

    model = ResNet2D(ResidualBlock2D, [2, 2, 2, 2], num_classes=6)
    model.load_state_dict(torch.load(
        'final_model_stft.pt', map_location='cpu'))
    model.eval()
    return model


def predict_signal(model, signal):
    dataset = AlignedEEGDataset(signal)
    dataloader = DataLoader(dataset)
    predictions = []

    with torch.no_grad():
        for x_feat in dataloader:
            logits, _ = model(x_feat)
            predicted_labels = torch.argmax(logits, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())

    return class_names[predictions[-1]] if predictions else 'UNKNOWN'
