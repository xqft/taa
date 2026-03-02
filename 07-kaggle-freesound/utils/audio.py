import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import tensorflow as tf


class Audio:
    def __init__(self, path, sample_rate=None, target_duration=None):
        self.path = Path(path)
        y, sr = self.audio_load()
        self.sr = sr
        self.y = y
        if target_duration:
            y = self._match_duration(y, sr, target_duration)
        self.duration = tf.shape(y)[0] / sr  # Duración del audio en segundos
        self.stft = tf.signal.stft(
            y[:, 0],
            frame_length=1024,
            frame_step=512,
            window_fn=tf.signal.hann_window,
            fft_length=1024,
        )
        self.espectograma = None

    def audio_load(self):
        audio_data = tf.io.read_file(
            str(self.path)
        )  # "returns a tensor with the entire contents of the input filename"
        waveform, sample_rate = tf.audio.decode_wav(audio_data)
        return waveform, sample_rate

    def log_magnitude_spectrogram(self):
        stft = self.stft
        # Obtengo magnitud de STFT
        spectrogram = tf.abs(stft)

        # Se define el banco de filtros a utilizar
        num_spectrogram_bins = spectrogram.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0, 4000, 80
        sample_rate = self.sr
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
        )

        # Se aplica el banco de filtros sobre el espectrograma
        mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )

        # Convertir el espectrograma a dB
        log_magnitude_spectrogram = tf.math.log(
            mel_spectrograms / tf.math.reduce_max(mel_spectrograms)
        )  # Agregar pequeño número para evitar log(0)
        return log_magnitude_spectrogram

    def print_audio_data(self):
        y = self.y
        sr = self.sr
        duration = self.duration
        if y is not None and sr is not None and duration is not None:
            print(f"Datos de audio: {y}")
            print(f"Frecuencia de muestreo: {sr}")
            print(f"Duración: {duration} segundos")

    def mostrar_espectograma(self):
        plt.figure(figsize=(10, 6))
        espectograma = self.log_magnitude_spectrogram()
        self.espectograma = espectograma
        plt.figure()
        plt.imshow(tf.transpose(espectograma).numpy(), aspect="auto", origin="lower")
        plt.title("Espectrograma de magnitud")
        nombre_sin_extension = self.path.stem
        plt.savefig(
            "espectrograma_" + nombre_sin_extension + ".png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()

    def _match_duration(self, y, sr, target_duration):
        current_duration = tf.shape(y)[0] / sr
        if current_duration < target_duration:
            y = np.pad(
                y, (0, int((target_duration - current_duration) * sr)), mode="constant"
            )
        else:
            y = y[: int(target_duration * sr)]
        return y

    def save_spectrogram(self, output_dir):
        plt.figure()
        espectograma = self.log_magnitude_spectrogram()
        self.espectograma = espectograma
        plt.imshow(tf.transpose(espectograma).numpy(), aspect="auto", origin="lower")
        plt.axis("off")
        plt.tight_layout(pad=0)
        filename = f"espectrograma_{self.path.stem}.png"
        plt.savefig(Path(output_dir) / filename, bbox_inches="tight", pad_inches=0)
        plt.close()

    def obtener_nombre(self):
        return self.name.stem

    def obtener_matriz_desde_imagen(self):
        nombre = self.name.stem
        imagen = Image.open("espectrograma_" + nombre + ".png")
        matrix = np.array(imagen)
        return matrix
