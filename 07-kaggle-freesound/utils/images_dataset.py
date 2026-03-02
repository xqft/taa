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
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from concurrent.futures import ThreadPoolExecutor
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def get_name_from_file_path(path):
    return list(map(lambda s: s.split(".")[0].split("/")[-1], path))


class ImagesDataset:
    def __init__(
        self, audio_dir, image_dir, labels_csv, image_size=(224, 224), batch_size=16
    ):
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.labels_csv = labels_csv
        self.image_size = image_size
        self.batch_size = batch_size
        self.labels = None
        self.mlb = MultiLabelBinarizer()
        self.augment = Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                PitchShift(min_semitones=-6, max_semitones=6, p=0.5),
                TimeStretch(min_rate=0.5, max_rate=1.5, p=0.5),
                Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
            ]
        )

    def load_data(self, labels=None):

        if labels is None:
            labels = pd.read_csv(self.labels_csv)
        else :
            labels = pd.read_csv(labels)

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
        labels = labels.set_index("fname")

        self.labels = labels
        return labels

    
    def get_noisy_dataset(self):
        tf.data.experimental.enable_debug_mode()

        labels = self.load_data("C:/Users/Valen/Desktop/Laboratorio-2-TAA/taa-2024-freesound-audio-tagging-v-1/train_noisy.csv")
        classes = self.mlb.classes_

        noisy_dataset = tf.keras.utils.image_dataset_from_directory(
            "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/noisy_dataset",
            labels=None,
            image_size=self.image_size,
            batch_size=None,
            shuffle=False,
        )

        noisy_fnames = get_name_from_file_path(noisy_dataset.file_paths)
        noisy_fnames = [file_path.split("\\")[-1].split(".")[0] for file_path in noisy_dataset.file_paths]

        noisy_labels = labels.loc[noisy_fnames]

        noisy_labels_tensor = tf.data.Dataset.from_tensor_slices(noisy_labels)

        # Add labels to dataset
        noisy_dataset = tf.data.Dataset.zip((noisy_dataset, noisy_labels_tensor))

        print(
            "Identified",
            labels.shape[1],
            "unique labels for",
            labels.shape[0],
            "files.",
        )

        # Shuffle and prefetch
        noisy_dataset = (
            noisy_dataset.shuffle(buffer_size=100)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        return noisy_dataset, classes


    def get_train_and_val_dataset(self):
        tf.data.experimental.enable_debug_mode()

        labels = self.load_data()
        classes = self.mlb.classes_

        train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
            self.image_dir,
            labels=None,
            validation_split=0.2,
            subset="both",
            image_size=self.image_size,
            batch_size=None,
            shuffle=False,  # this is so we can zip the labels in the correct order (alphanumerical)
        )

        train_fnames = get_name_from_file_path(train_dataset.file_paths)
        val_fnames = get_name_from_file_path(val_dataset.file_paths)
        
        train_fnames = [file_path.split("\\")[-1].split(".")[0] for file_path in train_dataset.file_paths]
        val_fnames = [file_path.split("\\")[-1].split(".")[0] for file_path in val_dataset.file_paths]

        train_labels = labels.loc[train_fnames]
        val_labels = labels.loc[val_fnames]

        train_labels_tensor = tf.data.Dataset.from_tensor_slices(train_labels)
        val_labels_tensor = tf.data.Dataset.from_tensor_slices(val_labels)

        # Add labels to dataset
        train_dataset = tf.data.Dataset.zip((train_dataset, train_labels_tensor))
        val_dataset = tf.data.Dataset.zip((val_dataset, val_labels_tensor))

        print(
            "Identified",
            labels.shape[1],
            "unique labels for",
            labels.shape[0],
            "files.",
        )

        # Shuffle and prefetch
        train_dataset = (
            train_dataset.shuffle(buffer_size=100)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        val_dataset = (
            val_dataset.shuffle(buffer_size=100)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        return train_dataset, val_dataset, classes
    
    def get_all_dataset(self):
        tf.data.experimental.enable_debug_mode()

        labels = self.load_data()
        classes = self.mlb.classes_

        dataset = tf.keras.utils.image_dataset_from_directory(
            self.image_dir,
            labels=None,
            image_size=self.image_size,
            batch_size=None,
            shuffle=False,
        )

        fnames = get_name_from_file_path(dataset.file_paths)
        fnames = [file_path.split("\\")[-1].split(".")[0] for file_path in dataset.file_paths]

        labels = labels.loc[fnames]

        labels_tensor = tf.data.Dataset.from_tensor_slices(labels)

        # Add labels to dataset
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

        return dataset, classes

    def augment_audio(self, file_path):
        # file_path = "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_folder_resampled_mixed/" + file_path + ".wav"
        samples, sample_rate = librosa.load(file_path, sr=None)
        augmented_samples = self.augment(samples=samples, sample_rate=sample_rate)
        augmented_path = os.path.join(
            os.path.dirname(file_path), Path(file_path).stem + "_augmented.wav"
        )
        sf.write(augmented_path, augmented_samples, sample_rate)
        return augmented_path

    def augment_dataset(self, dataset_directory):
        augmented_files = []
        print("Augmenting dataset...")
        print(dataset_directory)
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.augment_audio, os.path.join(root, file))
                for root, _, files in os.walk(dataset_directory)
                for file in files
                if file.endswith(".wav")
            ]
            for future in futures:
                augmented_files.append(future.result())
        return augmented_files

    def process_audios_logmel(self, input_dir=None, output_dir=None):

        if input_dir is None:
            input_dir = self.audio_dir
        if output_dir is None:
            output_dir = self.image_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

        # print(f"Archivos encontrados: {audio_files}")

        def stft_image(audio_file):
            try:
                # print(f"Procesando archivo: {input_dir}")
                y, sr = librosa.load(os.path.join(input_dir, audio_file), sr=None)

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

    def load_labels(self, file_path):
        if not hasattr(self, 'labels'):
            self.labels = self.load_data()
        base_name = os.path.basename(file_path)
        base_name = os.path.splitext(base_name)[0]
        return self.labels.loc[base_name].values

    def mixup_audio_and_labels(self, file1, file2, alpha=0.2):
        audio1, sr1 = librosa.load(file1, sr=None)
        audio2, sr2 = librosa.load(file2, sr=None)
        if sr1 != sr2:
            raise ValueError("Sampling rates are not equal.")
        labels1 = self.load_labels(file1)
        labels2 = self.load_labels(file2)

        min_length = min(len(audio1), len(audio2))

        # audio1, audio2 = audio1[:min_length], audio2[:min_length]

        lam = np.random.beta(alpha, alpha)

        # mixed_audio = lam * audio1 + (1 - lam) * audio2
        mixed_audio = lam * audio1[:min_length] + (1 - lam) * audio2[:min_length]

        if len(audio1) > min_length:
            mixed_audio = np.concatenate([mixed_audio, audio1[min_length:]])
        elif len(audio2) > min_length:
            mixed_audio = np.concatenate([mixed_audio, audio2[min_length:]])
        

        mixed_label = np.clip(labels1 + labels2, 0, 1)
        mixed_filename = f"{Path(file1).stem}_{Path(file2).stem}_mixed.wav"
        return mixed_audio, mixed_label, mixed_filename, sr1

    def mix_dataset(self, audio_dir, rounds=2):
        def is_mixed(filename):
            return "mixed" in filename
        
        filenames = [f for f in os.listdir(audio_dir) if not is_mixed(f)]
        np.random.shuffle(filenames)
        mixed_labels = []

        for _ in range(rounds):
            current_filenames = [f for f in os.listdir(audio_dir) if not is_mixed(f)]
            np.random.shuffle(current_filenames)
            new_mixed_files = []

            for i in range(0, len(current_filenames) - 1, 2):
                file1 = os.path.join(audio_dir, current_filenames[i])
                file2 = os.path.join(audio_dir, current_filenames[i + 1])
                mixed_audio, mixed_label, mixed_filename, sr = self.mixup_audio_and_labels(file1, file2)
                mixed_path = os.path.join(audio_dir, mixed_filename)
                sf.write(mixed_path, mixed_audio, sr)
                mixed_labels.append((mixed_filename, mixed_label))
                new_mixed_files.append(mixed_filename)

            filenames.extend(new_mixed_files)
            
        return mixed_labels
    
    def split_train_val(self, train_dir, val_dir, test_size=0.1, random_state=78557):
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        print("Dividiendo archivos de audio en entrenamiento y validación...")
        print(self.audio_dir)

        audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith(".wav")]

        train_files, val_files = train_test_split(
            audio_files, test_size=test_size, random_state=random_state
        )

        for file in train_files:
            shutil.move(
                os.path.join(self.audio_dir, file), os.path.join(train_dir, file)
            )

        for file in val_files:
            shutil.move(os.path.join(self.audio_dir, file), os.path.join(val_dir, file))

        print(f"Archivos de entrenamiento movidos a: {train_dir}")
        print(f"Archivos de validación movidos a: {val_dir}")

    def load_image(self, image_path, image_size=(224, 224)):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, image_size)
        return image

    def get_train_and_val_dataset_mixed_up(self):
        labels = self.load_data()
        classes = self.mlb.classes_

        print("Identified", labels.shape[1], "unique labels for", labels.shape[0], "files.")

        tf.data.experimental.enable_debug_mode()

        self.split_train_val("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_folder_resampled_mixed",
                             "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_folder_resampled_mixed")

        mixed_labels = self.mix_dataset("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_folder_resampled_mixed")
        # self.augment_dataset("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_folder_resampled_mixed")

        self.process_audios_logmel("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_folder_resampled_mixed",
                                   "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_preproc_logmel_resampled_mixed")
        self.process_audios_logmel("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_folder_resampled_mixed",
                                   "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_preproc_logmel_resampled_mixed")

        print("Columnas de mixed labels:", labels.columns)
        mixed_labels_df = pd.DataFrame(mixed_labels, columns=['fname', 'labels'])
        # Expande la lista de etiquetas en columnas separadas
        labels_expanded = pd.DataFrame(mixed_labels_df['labels'].tolist(), index=mixed_labels_df.index, columns=labels.columns)
        mixed_labels_df = pd.concat([mixed_labels_df[['fname']], labels_expanded], axis=1)
        mixed_labels_df['fname'] = mixed_labels_df['fname'].apply(lambda x: x.replace('.wav', '').replace('.png', ''))

        mixed_labels_df.set_index('fname', inplace=True)

        print(mixed_labels_df.tail(200))

        labels = pd.concat([labels, mixed_labels_df])

        # augmented_labels = labels.copy()
        # augmented_labels.index = augmented_labels.index + "_augmented"
        # labels = pd.concat([labels, augmented_labels])

        train_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(
                "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_preproc_logmel_resampled_mixed"
            )
        ]
        val_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(
                "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_preproc_logmel_resampled_mixed"
            )
        ]

        train_labels = labels.loc[train_files].values
        val_labels = labels.loc[val_files].values

        train_images = sorted(
            [
                os.path.join(
                    "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_preproc_logmel_resampled_mixed",
                    f,
                )
                for f in os.listdir(
                    "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_preproc_logmel_resampled_mixed"
                )
            ]
        )
        val_images = sorted(
            [
                os.path.join(
                    "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_preproc_logmel_resampled_mixed",
                    f,
                )
                for f in os.listdir(
                    "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_preproc_logmel_resampled_mixed"
                )
            ]
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.map(lambda x, y: (self.load_image(x, image_size=self.image_size), y))
        train_dataset = (
            train_dataset.batch(self.batch_size)
            .shuffle(len(train_images))
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.map(lambda x, y: (self.load_image(x, image_size=self.image_size), y))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        return train_dataset, val_dataset, classes

    def get_train_and_val_dataset_augmented(self):
            labels = self.load_data()
            classes = self.mlb.classes_

            print(
                "Identified",
                labels.shape[1],
                "unique labels for",
                labels.shape[0],
                "files.",
            )

            tf.data.experimental.enable_debug_mode()

            self.split_train_val("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_folder_resampled", "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_folder_resampled")

            self.augment_dataset("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_folder_resampled")

            self.process_audios_logmel("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_folder_resampled", "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_preproc_logmel_resampled")
            self.process_audios_logmel("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_folder_resampled", "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_preproc_logmel_resampled")

            augmented_labels = labels.copy()
            augmented_labels.index = augmented_labels.index + "_augmented"
            labels = pd.concat([labels, augmented_labels])

            train_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(
                    "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_preproc_logmel_resampled"
                )
            ]
            val_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(
                    "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_preproc_logmel_resampled"
                )
            ]

            train_labels = labels.loc[train_files].values
            val_labels = labels.loc[val_files].values

            train_images = sorted(
                [
                    os.path.join(
                        "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_preproc_logmel_resampled",
                        f,
                    )
                    for f in os.listdir(
                        "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/train_preproc_logmel_resampled"
                    )
                ]
            )
            val_images = sorted(
                [
                    os.path.join(
                        "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_preproc_logmel_resampled",
                        f,
                    )
                    for f in os.listdir(
                        "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/val_preproc_logmel_resampled"
                    )
                ]
            )

            train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            train_dataset = train_dataset.map(lambda x, y: (self.load_image(x), y))
            train_dataset = (
                train_dataset.batch(self.batch_size)
                .shuffle(len(train_images))
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            )

            val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
            val_dataset = val_dataset.map(lambda x, y: (self.load_image(x), y))
            val_dataset = val_dataset.batch(self.batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE
            )

            return train_dataset, val_dataset, classes
    
    def get_all_dataset_mixed(self):
            labels = self.load_data()
            classes = self.mlb.classes_

            print(
                "Identified",
                labels.shape[1],
                "unique labels for",
                labels.shape[0],
                "files.",
            )

            tf.data.experimental.enable_debug_mode()

            mixed_labels = self.mix_dataset("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/all_folder_resampled_mixed")

            self.process_audios_logmel("C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/all_folder_resampled_mixed", "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/all_preproc_logmel_resampled_mixed")
 

            print("Columnas de mixed labels:", labels.columns)
            mixed_labels_df = pd.DataFrame(mixed_labels, columns=['fname', 'labels'])
            labels_expanded = pd.DataFrame(mixed_labels_df['labels'].tolist(), index=mixed_labels_df.index, columns=labels.columns)
            mixed_labels_df = pd.concat([mixed_labels_df[['fname']], labels_expanded], axis=1)
            mixed_labels_df['fname'] = mixed_labels_df['fname'].apply(lambda x: x.replace('.wav', '').replace('.png', ''))

            mixed_labels_df.set_index('fname', inplace=True)

            labels = pd.concat([labels, mixed_labels_df])

            all_files = [
                os.path.splitext(f)[0]
                for f in os.listdir(
                    "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/all_preproc_logmel_resampled_mixed"
                )
            ]

            all_labels = labels.loc[all_files].values

            all_images = sorted(
                [
                    os.path.join(
                        "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/all_preproc_logmel_resampled_mixed",
                        f,
                    )
                    for f in os.listdir(
                        "C:/Users/Valen/Desktop/Laboratorio-2-TAA/data/all_preproc_logmel_resampled_mixed"
                    )
                ]
            )

            all_dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
            all_dataset = all_dataset.map(lambda x, y: (self.load_image(x), y))
            all_dataset = (
                all_dataset.batch(self.batch_size)
                .shuffle(len(all_images))
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            )

            return all_dataset, classes


    def gen_submission(
        self,
        model,
        test_image_dir="C:/Users/Valen/Desktop/Laboratorio-2-TAA/test_preproc_logmel_resampled/test_preproc_logmel_resampled",
        image_size=(224, 224),
        batch_size=16,
    ):
        self.load_data()
        print(self.mlb.classes_)
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
