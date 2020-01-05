import tensorflow as tf


def get_model():
    inputs = tf.keras.Input(shape=(36, 60, 1))
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(60, 3, input_shape=(36, 60, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(3),
        tf.keras.layers.Conv2D(120, 3, activation="tanh"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(150, 3),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1500, activation="softplus"),
    ])
    x = model(inputs)
    y = tf.keras.layers.Flatten()(inputs)
    res = x
   # res = tf.keras.layers.concatenate([x, y], axis=1)
    res = tf.keras.layers.Dense(1500, activation="relu")(res)
    res = tf.keras.layers.Dense(3)(res)
    model = tf.keras.Model(inputs, res)
    model.summary()
    return model

if __name__ == "__main__":
    get_model()