import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import random


df = None


# Colores predefinidos para elegir aleatoriamente
PREDEFINED_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]  # Rojo, Verde, Azul, Amarillo, Magenta

def random_color():
    """
    Selecciona aleatoriamente uno de los colores predefinidos.
    """
    return random.choice(PREDEFINED_COLORS)


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
    global df

    if df is None:
        messagebox.showwarning("Warning", "Please load a CSV file first.")
        return

    num_points = len(df)
    time = np.linspace(0, 10, num_points)

    max_amplitude = df.values.max() - df.values.min()
    offset = max_amplitude
    window_size = 512

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
                                linewidth=1, edgecolor='black', facecolor=random_color(), alpha=0.5)
        moving_rects.append(moving_rect)
        ax.add_patch(moving_rect)

        line, = ax.plot([], [], color='black', linewidth=0.5)
        lines.append(line)

        signal_line, = ax.plot([], [], color='black', linewidth=0.5)
        signal_lines.append(signal_line)

        ax.text(-0.2, i * offset, channel, verticalalignment='center', fontsize=8, color='blue')

    ax.set_xlim(0, 10)
    ax.set_ylim(-offset, offset * len(df.columns))
    ax.set_xlabel('Time (s)')
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks([])
    ax.grid(True)

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

                # Imprimir en consola la data que pasa por el cuadro
                print(f"Canal {df.columns[i]} - Data en cuadro:")
                print(df.iloc[start_idx:end_idx, i].to_list())

        for i, rect in enumerate(moving_rects):
            current_time = time[frame] - rect_duration
            rect.set_x(current_time)
            rect.set_facecolor(random_color())

        return lines + moving_rects + signal_lines

    step = 10
    ani = FuncAnimation(fig, update, frames=range(0, num_points, step), init_func=init, interval=1, repeat=False)
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
