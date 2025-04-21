import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from eeg_model_prediction import load_model, predict_signal


plt.rcParams.update({'font.family': 'Times New Roman'})

df = None
model = load_model() 

CLASS_COLORS = {
    'EEG': 'green',
    'Chewing': 'red',
    'Electrode pop': 'blue',
    'Eye movement': 'yellow',
    'Muscle': 'magenta',
    'Shiver': 'cyan'
}

def load_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    try:
        df = pd.read_csv(file_path)

        for item in tree.get_children():
            tree.delete(item)

        tree['columns'] = list(df.columns)
        tree['show'] = 'headings'

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        messagebox.showinfo("Success", "File loaded successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to read the file:\n{str(e)}")


def animate_signals():
    global df, model

    if df is None:
        messagebox.showwarning("Warning", "Please load a CSV file first.")
        return

    num_points = len(df)
    time = np.linspace(0, 10, num_points)

    max_amplitude = df.values.max() - df.values.min()
    offset = max_amplitude
    window_size = 512
    step_size = window_size
    
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.get_current_fig_manager().window.state('zoomed')

    lines = []
    moving_rects = []
    signal_lines = []

    sample_duration = 10 / num_points
    rect_duration = sample_duration * window_size

    scale_factor = 2.0

    for i, channel in enumerate(df.columns):
        rect_height = offset
        moving_rect = Rectangle((-0.5, i * offset - offset / 2), rect_duration, rect_height,
                                linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5)
        moving_rects.append(moving_rect)
        ax.add_patch(moving_rect)

        line, = ax.plot([], [], color='black', linewidth=0.5)
        lines.append(line)

        signal_line, = ax.plot([], [], color='black', linewidth=0.5)
        signal_lines.append(signal_line)

        ax.text(-0.5, i * offset, channel, verticalalignment='center', fontsize=20, color='black')

    ax.set_xlim(0, 10)
    ax.set_ylim(-offset, offset * len(df.columns))
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=20)
    ax.grid(True)

    legend_patches = [Rectangle((0, 0), 1, 1, color=color) for color in CLASS_COLORS.values()]
    ax.legend(legend_patches, CLASS_COLORS.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(CLASS_COLORS), fontsize=20)


    def init():
        for line in lines + signal_lines:
            line.set_data([], [])
        return lines + moving_rects + signal_lines

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(time[:frame], (df.iloc[:frame, i] * scale_factor) + i * offset)

        for i, signal_line in enumerate(signal_lines):
            start_idx = max(0, frame - window_size)
            end_idx = frame

            if start_idx < len(time) and end_idx <= len(time):
                time_segment = time[start_idx:end_idx]
                signal_segment = (df.iloc[start_idx:end_idx, i] * scale_factor) + i * offset
                signal_line.set_data(time_segment, signal_segment)

                segment_data = df.iloc[start_idx:end_idx, i].values
                prediction = predict_signal(model, segment_data)

                print(f"Canal {df.columns[i]} - Predicción: {prediction}")

                if prediction in CLASS_COLORS:
                    moving_rects[i].set_facecolor(CLASS_COLORS[prediction])
                    moving_rects[i].set_alpha(0.8)

        for i, rect in enumerate(moving_rects):
            current_time = time[frame] - rect_duration
            rect.set_x(current_time)

        return lines + moving_rects + signal_lines

    step = step_size
    ani = FuncAnimation(fig, update, frames=range(0, num_points, step), init_func=init, interval=100, repeat=False)
    plt.show()


def close_window():
    window.destroy()


window = tk.Tk()
window.title("CSV File Viewer")
window.geometry("800x500")

frame = tk.Frame(window)
frame.pack(fill=tk.BOTH, expand=True)

tree = ttk.Treeview(frame)

scrollbar_y = tk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

tree.configure(yscrollcommand=scrollbar_y.set)

scrollbar_x = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

tree.configure(xscrollcommand=scrollbar_x.set)

tree.pack(fill=tk.BOTH, expand=True)

button_frame = tk.Frame(window)
button_frame.pack(fill=tk.X, pady=10)

btn_load = tk.Button(button_frame, text="Load CSV File", command=load_file)
btn_load.pack(side=tk.LEFT, padx=10)

btn_animate = tk.Button(button_frame, text="Animate Signals", command=animate_signals)
btn_animate.pack(side=tk.LEFT, padx=10)

btn_exit = tk.Button(button_frame, text="Exit", command=close_window)
btn_exit.pack(side=tk.RIGHT, padx=10)

window.mainloop()
