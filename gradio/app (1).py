import gradio as gr
from PIL import Image
import os
import pickle
import random
import secrets
import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

torch.set_float32_matmul_precision('high')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

import TMIDIX
from x_transformer import TransformerWrapper, Decoder, AutoregressiveWrapper

from mido import MidiFile
import io

import tempfile
import torch
import TMIDIX
from midi2audio import FluidSynth
from pydub import AudioSegment

full_path_to_model_checkpoint = "model_checkpoint_9000_steps_1.4183_loss_0.6086_acc.pth"

# Принудительно используем CPU
device = torch.device("cpu")

# Используем float32 — стандартную и поддерживаемую точность на CPU
dtype = torch.float32

torch.set_float32_matmul_precision('high')

SEQ_LEN = 512

# Убираем use_flash_attn для CPU
model = TransformerWrapper(
    num_tokens=3088,
    max_seq_len=SEQ_LEN,
    attn_layers=Decoder(dim=512, depth=12, heads=10)
)

model = AutoregressiveWrapper(model)

state_dict = torch.load(full_path_to_model_checkpoint, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# Кастомные стили (оставляю без изменений)
custom_css = """
/* ... твой css ... */
html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    box-sizing: border-box;
}
body {
    background-color: #1e1e1e;
    font-family: "Inter", sans-serif;
    color: #e0e0e0;
    padding: 40px;
}
.gradio-container {
    width: 100%;
    max-width: 100%;
    margin: 0 auto;
    background: #2a2a2a;
    border: 1px solid #555555;
    border-radius: 10px;
    padding: 30px;
    box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.8);
    box-sizing: border-box;
}
.gradio-container .block,
.gradio-container .component,
.gradio-container .card,
.gradio-container .wrap,
.gradio-container .markdown,
.gradio-container .file,
.gradio-container .audio,
.gradio-container .output {
    background-color: #2a2a2a !important;
    border: 1px solid #555555 !important;
    border-radius: 6px;
    padding: 5px;
}
h1 {
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 20px;
}
button, select, input, textarea {
    border-radius: 6px;
    padding: 10px;
    border: 1px solid #555555;
    background: #3a3a3a;
    color: #e0e0e0;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
}
button:hover {
    background: #4a4a4a;
}
.slider, .dropdown, .checkbox {
    margin-bottom: 15px;
}
input[type="range"] {
    -webkit-appearance: none;
    width: 100%;
    margin: 6px 0;
    background: #3a3a3a;
    border-radius: 5px;
    height: 8px;
    border: 1px solid #555555;
}
input[type="range"]:focus {
    outline: none;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #555555;
    border: 2px solid #777777;
    cursor: pointer;
    margin-top: -6px;
}
input[type="range"]::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #555555;
    border: 2px solid #777777;
    cursor: pointer;
}
input[type="checkbox"] {
    accent-color: #555555;
}
.gradio-container input[type="checkbox"] ~ label,
.gradio-container .checkbox label {
    color: #e0e0e0 !important;
    cursor: pointer;
    user-select: none;
}
audio::-webkit-media-controls-play-button {
    filter: brightness(1.2);
    transform: scale(1.5);
}
audio {
    transform: scale(1.1);
}
"""

SOUNDFONT_PATH = "./FluidR3_GM.sf2"

def update_audio_choices(audio_files):
    if not audio_files:
        return gr.update(choices=[], value=None)
    else:
        return gr.update(choices=audio_files, value=audio_files[0])

def play_selected_audio(selected_audio):
    if not selected_audio:
        return None
    return selected_audio

def visualize_midi(midi_path):
    midi = MidiFile(midi_path)
    notes = []
    for track in midi.tracks:
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                channel = msg.channel if hasattr(msg, 'channel') else 0
                notes.append((time, msg.note, channel))
    if not notes:
        return None
    
    times = [n[0] for n in notes]
    pitches = [n[1] for n in notes]
    channels = [n[2] for n in notes]
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'brown']
    
    plt.figure(figsize=(14, 4))
    plt.scatter(times, pitches, c=[colors[ch % len(colors)] for ch in channels], s=10)
    plt.xlabel('Time (ticks)')
    plt.ylabel('Pitch (MIDI note number)')
    plt.title('MIDI Note Visualization')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return Image.open(buf)

def generate_music_gradio(first_note_instrument, add_drums,
                          num_tokens, num_batches, temperature, render_MIDI_to_Audio):
    out_dir = tempfile.mkdtemp()
    instruments_list = ["Piano", "Guitar", "Bass", "Violin", "Cello", "Harp",
                        "Trumpet", "Sax", "Flute", "Choir", "Organ"]
    first_note_instrument_number = instruments_list.index(first_note_instrument)
    drumsp = 3074 if add_drums else 3073
    outy = [3087, drumsp, 3075 + first_note_instrument_number]

    inp = [outy] * num_batches
    inp = torch.LongTensor(inp).to(device)

    with torch.no_grad():
        out = model.generate(inp, num_tokens, temperature=temperature, return_prime=True, verbose=True)
    out0 = out.tolist()

    midi_files = []
    audio_files = []

    for i in range(num_batches):
        out1 = out0[i]
        song_f = []
        time = 0; dur = 0; vel = 90; pitch = 0; channel = 0
        for ss in out1:
            if ss > 0 and ss < 256:
                time += ss * 8
            if ss >= 256 and ss < 1280:
                dur = ((ss - 256) // 8) * 32
                vel = (((ss - 256) % 8) + 1) * 15
            if ss >= 1280 and ss < 2816:
                channel = (ss - 1280) // 128
                pitch = (ss - 1280) % 128
                song_f.append(['note', time, dur, channel, pitch, vel])

        base_fname = os.path.join(out_dir, f"composition_{i}")
        midi_fname = base_fname + ".mid"
        TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f, output_file_name=base_fname)

        midi_files.append(midi_fname)

        if render_MIDI_to_Audio:
            wav_fname = base_fname + ".wav"
            try:
                FluidSynth(SOUNDFONT_PATH).midi_to_audio(midi_fname, wav_fname)
                mp3_fname = base_fname + ".mp3"
                sound = AudioSegment.from_wav(wav_fname)
                sound.export(mp3_fname, format="mp3", bitrate="128k")
                audio_files.append(mp3_fname)
            except Exception as e:
                print("Ошибка при конвертации в аудио:", e)

    # Визуализация первого MIDI файла
    if midi_files:
        img_buf = visualize_midi(midi_files[0])
    else:
        img_buf = None

    return midi_files, audio_files, img_buf

iface = gr.Blocks(css=custom_css)

with iface:
    gr.HTML("""
        <style>
          body {
            overflow-y: auto !important;
              }
      html, body, #root {
        height: auto !important;
        min-height: 100vh;
      }
    </style>
        """)

    gr.Markdown("<h1 style='text-align: center;'>Pocket music generator</h1>")

    first_note_instrument = gr.Dropdown(
        ["Piano", "Guitar", "Bass", "Violin", "Cello", "Harp",
         "Trumpet", "Sax", "Flute", "Choir", "Organ"],
        label="Выберите инструмент"
    )
    add_drums = gr.Checkbox(label="Добавить барабаны")
    num_tokens = gr.Slider(30, 2048, step=3, label="Число токенов", value=300)
    num_batches = gr.Slider(1, 16, step=1, label="Число треков для генерации", value=1)
    temperature = gr.Slider(0.1, 1.0, step=0.1, label="Температура", value=0.7)
    render_audio = gr.Checkbox(label="Конвертировать в MP3", value=True)

    generate_btn = gr.Button("Сгенерировать музыку")

    audio_files_state = gr.State([])
    audio_selector = gr.Dropdown(choices=[], label="Выберите аудиофайл для прослушивания", allow_custom_value=True)
    audio_output = gr.Audio(label="Прослушивание", format="mp3")
    midi_output = gr.File(label="Скачать MIDI")

    midi_visualization = gr.Image(label="Визуализация MIDI", interactive=False)

    generate_btn.click(
        generate_music_gradio,
        inputs=[first_note_instrument, add_drums, num_tokens, num_batches, temperature, render_audio],
        outputs=[midi_output, audio_files_state, midi_visualization]
    ).then(
        update_audio_choices,
        inputs=[audio_files_state],
        outputs=[audio_selector]
    )

    audio_selector.change(
        play_selected_audio,
        inputs=[audio_selector],
        outputs=[audio_output]
    )

iface.launch(debug=True)
