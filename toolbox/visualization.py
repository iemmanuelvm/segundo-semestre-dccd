import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from matplotlib.patches import Rectangle

# Dictionary of signals for easier selection and labeling
SIGNALS_DICT = {
    0: {"label": "EEG"},
    1: {"label": "EEG + EOG"},
    2: {"label": "EEG + EMG"}
}

class MultiPlotter():
    def __init__(self, num_segments=5, win_size=512, seconds=5, FIG_SIZE=(12, 8), title=""):
        """
        Parameters:
          - num_segments: number of subplots (windows) to display.
          - win_size: number of samples corresponding to 1 second (e.g., 512).
          - seconds: duration (in seconds) of each subplot.
          - FIG_SIZE: size of the figure.
        """
        self.num_segments = num_segments
        self.win_size = win_size          # samples per second
        self.seconds = seconds            # seconds to display per plot
        self.total_win = win_size * seconds  # total samples to display (e.g., 512*5 = 2560)

        # Interactive mode
        plt.ion()

        # Create subplots: each will display 5 seconds of signal
        self.fig, self.axes = plt.subplots(num_segments, 1, figsize=FIG_SIZE, sharex=True)
        if num_segments == 1:
            self.axes = [self.axes]

        self.lines = []
        self.texts = []       # Texts to be shown in the corner (if needed)
        self.rectangles = [None] * self.num_segments  # Rectangle for each subplot
        # List to store text placed inside the rectangle
        self.rect_texts = [None] * self.num_segments

        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], lw=2)
            self.lines.append(line)
            ax.set_ylim(-10, 10)
            ax.set_ylabel("Amplitude")
            # X-axis goes from 0 to 'seconds' (0 to 5 s)
            ax.set_xlim(0, seconds)
            ticks = np.linspace(0, seconds, num=6)
            tick_labels = np.round(ticks, 2)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
            # Optional text in the upper right corner
            text = ax.text(0.95, 0.9, "", transform=ax.transAxes,
                           horizontalalignment='right', fontsize=12, color='red')
            self.texts.append(text)

        self.axes[-1].set_xlabel("Time (seconds)")
        self.fig.suptitle(title)
        plt.tight_layout()
        self.fig.show()

    def update(self, new_signals, predictions, actual_labels, current_block, title):
        """
        new_signals: array of shape (num_segments, total_win) with accumulated data.
        predictions: array of predictions (one per subplot) made on the current block (win_size samples).
        actual_labels: list of actual labels (the signal from which each window was extracted).
        current_block: current block (in seconds) to position the rectangle.
        title: overall title for the figure.
        """
        # Define colors for each prediction
        color_map = {
            0: "green",  # EEG
            1: "blue",   # EEG + EOG
            2: "red"     # EEG + EMG
        }
        default_color = "gray"  # for unknown predictions

        for i in range(self.num_segments):
            signal = new_signals[i].squeeze()
            # Create a time vector from 0 to 'seconds' with total_win points.
            t = np.linspace(0, self.seconds, self.total_win)
            self.lines[i].set_data(t, signal)
            # Convert numeric prediction to label
            if predictions[i] in color_map:
                pred_label = SIGNALS_DICT[predictions[i]]["label"]
                rect_color = color_map[predictions[i]]
            else:
                pred_label = "Unknown"
                rect_color = default_color

            # Remove the previous rectangle, if it exists
            if self.rectangles[i] is not None:
                self.rectangles[i].remove()

            # Draw a rectangle over the current block (each block corresponds to 1 second)
            rect = Rectangle((current_block - 1, -10), 1, 20, linewidth=1,
                             edgecolor=rect_color, facecolor=rect_color, alpha=0.3)
            self.axes[i].add_patch(rect)
            self.rectangles[i] = rect

            # Remove previous text inside the rectangle, if exists
            if self.rect_texts[i] is not None:
                self.rect_texts[i].remove()

            # Position the text at the top of the rectangle
            x_center = current_block - 0.5  # center of the rectangle
            y_top = 10 - 0.5                # a little below the top edge (with 0.5 margin)
            rect_text = self.axes[i].text(x_center, y_top, f"Prediction: {pred_label}",
                                          horizontalalignment="center", verticalalignment="top",
                                          fontsize=10, color="white", weight="bold")
            self.rect_texts[i] = rect_text

        self.fig.suptitle(title)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)


# Example usage in a simulation (similar to the animate_random_segments method)
def animate_random_segments(model, EEG, EEG_EOG, EEG_EMG, win_size=512, seconds=5, num_segments=5, cycles=3):
    """
    Simulate the arrival of data and update the display.
    """
    # Group the signals in a dictionary
    signals = {
        0: {"data": EEG,     "label": "EEG"},
        1: {"data": EEG_EOG, "label": "EEG + EOG"},
        2: {"data": EEG_EMG, "label": "EEG + EMG"}
    }

    total_win = win_size * seconds  # total samples in each window (e.g., 2560)
    plotter = MultiPlotter(num_segments=num_segments, win_size=win_size, seconds=seconds)

    # Define the shift (25% of win_size)
    shift = int(0.25 * win_size)  # 128 samples if win_size=512

    for cycle in range(cycles):
        segments = []         # each element is the complete window (2560 samples)
        actual_labels = []    # actual label for each window
        # For each subplot, select a random signal and extract a complete window of total_win samples
        for i in range(num_segments):
            chosen_key = random.choice(list(signals.keys()))
            actual_labels.append(signals[chosen_key]["label"])
            sig_data = signals[chosen_key]["data"]
            total_length = len(sig_data)
            max_start = total_length - total_win
            if max_start <= 0:
                raise ValueError(f"The signal {signals[chosen_key]['label']} does not have enough samples.")
            start_index = random.randint(0, max_start)
            segment = sig_data[start_index: start_index + total_win]
            segments.append(segment)
        segments = np.array(segments)  # shape: (num_segments, total_win)

        # Simulate data arrival every 25% of 1 second (shift of 128 samples)
        for current_index in range(win_size, total_win + 1, shift):
            # Build the accumulated data vector for each subplot:
            accumulated_signals = []
            for i in range(num_segments):
                # Accumulate data up to the current index
                accumulated = segments[i][:current_index]
                # Pad the rest of the window with np.nan to avoid plotting empty data
                pad = np.full(total_win - current_index, np.nan)
                full_signal = np.concatenate([accumulated, pad])
                accumulated_signals.append(full_signal)
            accumulated_signals = np.array(accumulated_signals)

            # For prediction, use a window of win_size samples
            current_window_data = segments[:, current_index - win_size: current_index]
            current_window_tensor = torch.tensor(current_window_data, dtype=torch.float32).unsqueeze(1)

            model.eval()
            with torch.no_grad():
                # Assume the model accepts tensors of shape [batch, 1, win_size]
                output = model(current_window_tensor.permute(0, 2, 1))
                predictions = torch.argmax(output, dim=1).cpu().numpy()

            # Calculate current time in seconds (win_size samples = 1 s)
            current_time = current_index / win_size
            # Update the display by drawing a rectangle indicating the current window.
            plotter.update(accumulated_signals, predictions, actual_labels,
                           current_block=current_time,
                           title=f"Random Signals - Cycle {cycle+1}, Time {current_time:.2f} s")
            # Pause to simulate new data arrival
            time.sleep(0.25)
