import tensorflow as tf

# Different architectures to test:

def arch_a0():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(8,8,6)),
        tf.keras.layers.Conv2D(1, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

def arch_a1_d0():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(8,8,6)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="swish"),
        tf.keras.layers.Dense(1)
    ])

def arch_a1_d1():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(8,8,6)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="swish"),
        tf.keras.layers.Dense(128, activation="swish"),
        tf.keras.layers.Dense(1)
    ])


def arch_a1_c0():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(8,8,6)),
        tf.keras.layers.Conv2D(6,3, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(5,7, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(4,17, padding="same", activation="swish"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="swish"),
        tf.keras.layers.Dense(1)
    ])

def arch_a1_c1():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(8,8,6)),
        tf.keras.layers.Conv2D(6,3, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(6,3, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(6,3, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(6,3, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(6,3, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(6,3, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(6,3, padding="same", activation="swish"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="swish"),
        tf.keras.layers.Dense(1)
    ])

def arch_a1_c2():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(8,8,6)),
        tf.keras.layers.Conv2D(6,7, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(6,7, padding="same", activation="swish"),
        tf.keras.layers.Conv2D(6,7, padding="same", activation="swish"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="swish"),
        tf.keras.layers.Dense(1)
    ])