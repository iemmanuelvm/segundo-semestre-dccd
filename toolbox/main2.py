from eeg_model_prediction import load_model, predict_signal
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import matplotlib
matplotlib.use('TkAgg')

app = FastAPI()

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


@app.get("/", response_class=HTMLResponse)
def read_root():
    global df
    if df is None:
        return """
        <h1>No se ha cargado ningún archivo CSV todavía.</h1>
        <p>Usa un cliente (Postman, cURL, etc.) para enviar un archivo CSV a <code>/upload-file</code>.</p>
        """
    else:
        return """
        <h1>Archivo CSV cargado correctamente.</h1>
        <p>Ya se ha graficado con éxito.</p>
        """


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    global df
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        graficar(df)
        return {"filename": file.filename, "message": "Archivo CSV cargado y gráfica finalizada."}

    except Exception as e:
        return {"error": str(e)}


def graficar(dataframe: pd.DataFrame):
    num_points = len(dataframe)
    time = np.linspace(0, 10, num_points)

    max_amplitude = dataframe.values.max() - dataframe.values.min()
    offset = max_amplitude
    window_size = 512
    step_size = window_size

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.canvas.manager.toolbar.pack_forget()
    fig.canvas.manager.window.overrideredirect(True)
    fig.canvas.manager.window.geometry("480x320+0+0")

    lines = []
    moving_rects = []
    signal_lines = []

    sample_duration = 10 / num_points
    rect_duration = sample_duration * window_size

    scale_factor = 2.0

    for i, channel in enumerate(dataframe.columns):
        rect_height = offset
        moving_rect = Rectangle(
            (-0.5, i * offset - offset / 2),
            rect_duration,
            rect_height,
            linewidth=1,
            edgecolor='black',
            facecolor='gray',
            alpha=0.5
        )
        moving_rects.append(moving_rect)
        ax.add_patch(moving_rect)

        line, = ax.plot([], [], color='black', linewidth=0.5)
        lines.append(line)

        signal_line, = ax.plot([], [], color='black', linewidth=0.5)
        signal_lines.append(signal_line)

        ax.text(-0.5, i * offset, channel, verticalalignment='center',
                fontsize=10, color='black')

    ax.set_xlim(0, 10)
    ax.set_ylim(-offset, offset * len(dataframe.columns))
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(True)

    legend_patches = [Rectangle((0, 0), 1, 1, color=color)
                      for color in CLASS_COLORS.values()]
    ax.legend(
        legend_patches,
        CLASS_COLORS.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        fontsize=10
    )

    def init():
        for line in lines + signal_lines:
            line.set_data([], [])
        return lines + moving_rects + signal_lines

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(
                time[:frame], (dataframe.iloc[:frame, i] * scale_factor) + i * offset)

        for i, signal_line in enumerate(signal_lines):
            start_idx = max(0, frame - window_size)
            end_idx = frame

            if start_idx < len(time) and end_idx <= len(time):
                time_segment = time[start_idx:end_idx]
                signal_segment = (
                    dataframe.iloc[start_idx:end_idx, i] * scale_factor) + i * offset
                signal_line.set_data(time_segment, signal_segment)

                segment_data = dataframe.iloc[start_idx:end_idx, i].values
                prediction = predict_signal(model, segment_data)
                print(
                    f"Canal {dataframe.columns[i]} - Predicción: {prediction}")

                if prediction in CLASS_COLORS:
                    moving_rects[i].set_facecolor(CLASS_COLORS[prediction])
                    moving_rects[i].set_alpha(0.8)

        for i, rect in enumerate(moving_rects):
            current_time = time[frame] - rect_duration
            rect.set_x(current_time)

        return lines + moving_rects + signal_lines

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, num_points, step_size),
        init_func=init,
        interval=100,
        repeat=False
    )

    ax_button = fig.add_axes([0.9, 0.02, 0.08, 0.05])

    btn_ok = Button(ax_button, 'OK')

    def cerrar_figura(event):
        plt.close(fig)

    btn_ok.on_clicked(cerrar_figura)

    plt.show()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
