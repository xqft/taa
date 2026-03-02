import sys

sys.path.append("../")

import pandas as pd
import tensorflow as tf
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy
import shutil
from utils.audio import Audio

from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from concurrent.futures import ThreadPoolExecutor
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.model_selection import StratifiedShuffleSplit
from spec_augment import SpecAugment


def get_name_from_file_path(path):
    return list(map(lambda s: s.split(".")[0].split("/")[-1], path))


class ImagesDataset:
    def __init__(
        self,
        audio_dir,
        test_audio_dir,
        labels_csv,
        image_size=(224, 224),
        batch_size=32,
    ):
        self.audio_dir = audio_dir
        self.test_audio_dir = test_audio_dir

        self.labels_csv = labels_csv
        self.image_size = image_size
        self.batch_size = batch_size

        self.train_audio_dir = "train_" + audio_dir
        self.val_audio_dir = "val_" + audio_dir
        self.train_audio_augmented_dir = self.train_audio_dir + "_augmented"
        self.train_audio_split_dir = self.train_audio_dir + "_split"
        self.train_audio_specaugment_dir = self.train_audio_dir + "_specaugment"

        self.mlb = MultiLabelBinarizer()
        self.augment = Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
            ]
        )

    def load_data(self):
        labels = pd.read_csv(self.labels_csv)
        labels["labels"] = labels["labels"].apply(lambda x: x.split(","))
        labels = pd.concat(
            [
                labels[["fname"]],
                pd.DataFrame(
                    self.mlb.fit_transform(labels["labels"]), columns=self.mlb.classes_
                ),
            ],
            axis=1,
        )
        # Remove .wav extension from file name
        labels["fname"] = labels["fname"].apply(lambda string: string.split(".")[0])
        # this will sort labels by the filename, alphanumerically, and drop the "fname" column.
        labels = labels.set_index("fname", drop=True)

        return labels

    def get_image_dataset(self, image_dir, labels):
        tf.data.experimental.enable_debug_mode()

        dataset = tf.keras.utils.image_dataset_from_directory(
            image_dir,
            labels=None,
            image_size=self.image_size,
            batch_size=None,
            shuffle=False,
        )

        fnames = get_name_from_file_path(dataset.file_paths)

        labels = labels.loc[fnames].to_numpy()

        labels_tensor = tf.data.Dataset.from_tensor_slices(labels)

        dataset = tf.data.Dataset.zip((dataset, labels_tensor))

        print(
            "Identified",
            labels.shape[1],
            "unique labels for",
            labels.shape[0],
            "files.",
        )

        # Shuffle and prefetch
        dataset = (
            dataset.shuffle(buffer_size=100)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        return dataset, labels

    def _augment_audio(self, file_path):
        samples, sample_rate = librosa.load(file_path, sr=None)
        augmented_samples = self.augment(samples=samples, sample_rate=sample_rate)
        augmented_path = os.path.join(
            os.path.dirname(file_path), Path(file_path).stem + "_augmented.wav"
        )
        sf.write(augmented_path, augmented_samples, sample_rate)
        return augmented_path

    def augment_audios_train(self, audio_dir=None):
        print("Augmenting train audios")

        if not hasattr(self, "train_audio_dir"):
            raise Exception("you need to call self.split_train_val() first")

        if audio_dir == None:
            audio_dir = self.train_audio_dir

        train_audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

        if not os.path.exists(self.train_audio_augmented_dir):
            os.makedirs(self.train_audio_augmented_dir)

        for file in train_audio_files:
            shutil.copyfile(
                os.path.join(audio_dir, file),
                os.path.join(self.train_audio_augmented_dir, file),
            )

        with tqdm_joblib(tqdm(total=len(train_audio_files))):
            Parallel(n_jobs=cpu_count() - 1)(
                delayed(self._augment_audio)(
                    os.path.join(self.train_audio_augmented_dir, file),
                )
                for file in train_audio_files
            )

        self.augmented_audios = True

    def _split_audio(self, origin_path, dest_dir, duration):
        data, sample_rate = sf.read(origin_path)
        sample_count = duration * sample_rate

        for i in range(0, len(data), sample_count):
            if len(data) - i < 2048:  # min samples for fft
                break
            split_path = os.path.join(
                dest_dir, Path(origin_path).stem + f"_{i//sample_count}.wav"
            )
            sf.write(split_path, data[i : i + sample_count], sample_rate)

    def split_audios_train(self, length, audio_dir=None):
        if not hasattr(self, "train_audio_dir"):
            raise Exception("you need to call self.split_train_val() first")

        if audio_dir == None:
            audio_dir = self.train_audio_dir

        train_audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

        if not os.path.exists(self.train_audio_split_dir):
            os.makedirs(self.train_audio_split_dir)

        with tqdm_joblib(tqdm(total=len(train_audio_files))):
            Parallel(n_jobs=cpu_count() - 1)(
                delayed(self._split_audio)(
                    os.path.join(self.train_audio_dir, file),
                    self.train_audio_split_dir,
                    length,
                )
                for file in train_audio_files
            )

    def process_audios_logmel(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

        # print(f"Archivos encontrados: {audio_files}")

        def stft_image(audio_file):
            try:
                # print(f"Procesando archivo: {input_dir}")
                y, sr = librosa.load(os.path.join(input_dir, audio_file), sr=None)
                # y = librosa.effects.trim(y, top_db=1)

                if len(y) <= 2048:
                    return

                y = y[
                    : int(5.0 * sr)
                ]  # 5.0 segundos multiplicados por la frecuencia de muestreo

                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=256, window=scipy.signal.windows.hann
                )
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                plt.figure()
                librosa.display.specshow(
                    log_mel_spec, sr=sr, x_axis="time", y_axis="mel"
                )
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(
                    os.path.join(output_dir, audio_file.split(".")[0]),
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()
                # clear_output(wait=True)
                # print("Procesando archivo", i, "de", len(audio_files), end="\r")
                # print(f"Espectrograma log-mel guardado en: {output_path}")

            except Exception as e:
                print(f"Error procesando el archivo {input_dir}: {e}")

        with tqdm_joblib(tqdm(total=len(audio_files))):
            Parallel(n_jobs=cpu_count() - 1)(
                delayed(stft_image)(audio_file) for audio_file in audio_files
            )

    def split_train_val(self, test_size=0.2, random_state=78557):
        if not os.path.exists(self.train_audio_dir):
            os.makedirs(self.train_audio_dir)
        if not os.path.exists(self.val_audio_dir):
            os.makedirs(self.val_audio_dir)

        print("Dividiendo archivos de audio en entrenamiento y validación...")

        audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith(".wav")]

        train_files, val_files = train_test_split(
            audio_files,
            test_size=test_size,
            random_state=random_state,
        )

        for file in train_files:
            shutil.copyfile(
                os.path.join(self.audio_dir, file),
                os.path.join(self.train_audio_dir, file),
            )

        for file in val_files:
            shutil.copyfile(
                os.path.join(self.audio_dir, file),
                os.path.join(self.val_audio_dir, file),
            )

        print(
            f"{len(train_files)} archivos de entrenamiento movidos a: {self.train_audio_dir}"
        )
        print(
            f"{len(val_files)} archivos de validación movidos a: {self.val_audio_dir}"
        )

    def get_train_and_val_dataset(
        self, use_augmented_audios=False, use_split_audios=False
    ):
        if not os.path.exists(self.train_audio_dir):
            raise Exception("you need to call self.split_train_val() first")

        labels = self.load_data()
        classes = self.mlb.classes_

        tf.data.experimental.enable_debug_mode()

        if use_augmented_audios:
            if not os.path.exists(self.train_audio_augmented_dir):
                raise Exception("you need to call self.augment_audios_train() first")
            self.train_image_dir = self.train_audio_augmented_dir + "_spectrograms"
            augmented_labels = labels.copy()
            augmented_labels.index = augmented_labels.index + "_augmented"
            labels = pd.concat([labels, augmented_labels])
            train_audio_dir = self.train_audio_augmented_dir
        elif use_split_audios:
            if not os.path.exists(self.train_audio_split_dir):
                raise Exception("you need to call self.split_audios_train() first")
            self.train_image_dir = self.train_audio_split_dir + "_spectrograms"

            train_audio_split_files = [
                f for f in os.listdir(self.train_audio_split_dir) if f.endswith(".wav")
            ]
            split_labels = labels.copy()
            split_labels.index = split_labels.index + "_0"
            for file in train_audio_split_files:
                split_labels.loc[file.split(".")[0]] = labels.loc[file.split("_")[0]]
            labels = pd.concat([labels, split_labels])
            train_audio_dir = self.train_audio_split_dir
        else:
            self.train_image_dir = self.train_audio_dir + "_spectrograms"
            train_audio_dir = self.train_audio_dir

        print(self.train_image_dir)
        self.val_image_dir = self.val_audio_dir + "_spectrograms"

        if not os.path.exists(self.train_image_dir):
            print("No se encontraron espectrogramas de entrenamiento. Generando..")
            self.process_audios_logmel(train_audio_dir, self.train_image_dir)
        if not os.path.exists(self.val_image_dir):
            print("No se encontraron espectrogramas de validacion. Generando..")
            self.process_audios_logmel(self.val_audio_dir, self.val_image_dir)

        train_dataset, train_labels = self.get_image_dataset(
            self.train_image_dir, labels
        )
        val_dataset, val_labels = self.get_image_dataset(self.val_image_dir, labels)

        self.train_labels = train_labels
        self.val_labels = val_labels

        return train_dataset, val_dataset, classes

    def gen_submission(
        self,
        model,
        test_image_dir="test_preproc_logmel",
        image_size=(200, 200),
        batch_size=32,
    ):
        tf.data.experimental.enable_debug_mode()

        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_image_dir,
            labels=None,
            image_size=image_size,
            batch_size=None,
            shuffle=False,
        )
        test_fnames = test_dataset.file_paths

        # Batch and prefetch
        test_dataset = test_dataset.batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        # Predict
        test_preds = model.predict(test_dataset)

        submission = pd.DataFrame(test_preds, columns=self.mlb.classes_)
        fnames = list(
            map(lambda fname: fname + ".wav", get_name_from_file_path(test_fnames))
        )
        submission.insert(loc=0, column="fname", value=fnames)
        submission.to_csv("submission.csv", index=False)

        return submission
