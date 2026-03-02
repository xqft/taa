import importlib
import sys

sys.path.append("../utils")

import tensorflow as tf
import keras_tuner as kt

import lwlrap
from lwlrap import LwLrap

importlib.reload(lwlrap)
from spec_augment import SpecAugment
from specaugment_wrapper import MaybeSpecAugment

num_classes = 80


def build_mobilenet(
    dropout_rate=0.3,
    learning_rate=1e-3,
    fine_tune=False,
    train_upto_layer=8 * 2 + 5,
    image_size=(224, 224),
    specaugment=True,
):
    if not fine_tune:
        # Define layers
        preproc = tf.keras.Sequential()
        if specaugment:
            spec_augment = MaybeSpecAugment(
                freq_mask_param=5,
                time_mask_param=10,
                n_freq_mask=5,
                n_time_mask=3,
                mask_value=-100,
            )
            preproc.add(spec_augment)
        preproc.add(tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1))
        avg = tf.keras.layers.GlobalAveragePooling2D()
        predict = tf.keras.layers.Dense(num_classes, activation="sigmoid")

        # Define model
        input = tf.keras.Input(shape=image_size + (3,))
        x = preproc(input)

        base_model = tf.keras.applications.MobileNet(
            input_shape=image_size + (3,),
            include_top=False,
            input_tensor=x,
            dropout=dropout_rate,
        )
        for layer in base_model.layers:
            layer.trainable = False

        x = avg(base_model.layers[-1].output)
        output = predict(x)
        model = tf.keras.Model(input, output)
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        # Load pre-trained model and freeze
        model = tf.keras.models.load_model(
            "baseline/best_mobilenetv2.keras",
            custom_objects={"lwlrap": LwLrap(num_classes)},
        )
        for layer in model.layers[: len(model.layers) - train_upto_layer]:
            layer.trainable = False
        for layer in model.layers[len(model.layers) - train_upto_layer :]:
            layer.trainable = True
        # Set lower learning rate as to not overfit
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate / 100)

    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=[LwLrap(num_classes)]
    )
    return model
