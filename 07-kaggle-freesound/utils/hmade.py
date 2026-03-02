import tensorflow as tf
import keras_tuner as kt
def ConvBlock(x, filters, downsample=False):
    residual = x

    convolution_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()])
    convolution_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, (2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()])
    convolution_full = tf.keras.Sequential([
                convolution_1,
                convolution_2,])
    x = convolution_full(x)
    return x

def create_model(num_classes=80,optimizer='Nadam',learning_rate=1e-3,image_size=(224, 224)):
    inputs = tf.keras.Input(shape=image_size + (3,))
    preproc = tf.keras.Sequential([
            tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)])
    predict = tf.keras.layers.Dense(num_classes, activation="sigmoid")
    
    x = preproc(inputs)
    x = ConvBlock(x, 64)
    x = ConvBlock(x, 128, downsample=True)
    x= tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = predict(x)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[LwLrap(num_classes)])
    return model