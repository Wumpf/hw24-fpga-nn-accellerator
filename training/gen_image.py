import os
import shutil
import rerun as rr

import numpy as np
from fxpmath import Fxp
import tensorflow as tf

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense

def load_ascii(name):
    with open(f"output/{name}.txt", "r") as f:
        floats = np.array([float(int(x, 16)) for x in f.read().strip().split(" ")])
        floats -= 127
        floats /= 128.0
        return floats
    
def load_ascii_reshape(name, x, y):
    with open(f"output/{name}.txt", "r") as f:
        floats = np.array([float(int(x, 16)) for x in f.read().strip().split(" ")])
        floats -= 127
        floats /= 128.0
        return floats.reshape((x, y))

def load_weights_and_bises():
    return {
        "dense": (load_ascii_reshape("dense_weights", 64, 64), load_ascii("dense_biases")),
        "dense_1": (load_ascii_reshape("dense_1_weights", 64, 64), load_ascii("dense_1_biases")),
        "dense_2": (load_ascii_reshape("dense_2_weights", 64, 64), load_ascii("dense_2_biases")),
        "dense_3": (load_ascii_reshape("dense_3_weights", 64, 3), load_ascii("dense_3_biases"))
    }

def set_weights(model, layers):
    # update model with fixed point and evaluate
    for layer, values in layers.items():
        model.get_layer(layer).set_weights(values)

# List all available devices
devices = tf.config.list_physical_devices()
print(f"Devices: {devices}")

rr.init("test_ascii", spawn=True)

width, height = 64, 64

# Prepare input data (X): Normalized pixel coordinates
x_coords = np.arange(width).reshape(-1, 1) / width
y_coords = np.arange(height).reshape(-1, 1) / height

def x_training_data_rnd_sin_cos_biased(x_coords, y_coords, embedding_size):
    seed = 127
    np.random.seed(seed)
    rnd = np.random.random_integers(0, 0xFFFFFF, embedding_size)
    return np.array([np.array([[np.sin(x * rnd[i] + y), np.cos(y * rnd[embedding_size-1-i] + x)] for i in range(embedding_size)]).flatten() for y in y_coords for x in x_coords])

embedding_size = 32
layer_size = 64

X = x_training_data_rnd_sin_cos_biased(x_coords, y_coords, embedding_size)

rr.log("log", rr.TextLog(str(X.shape)))

print(f"X.shape : {X.shape}")

model = Sequential([
    Dense(layer_size, input_dim=embedding_size*2, activation='relu'),
    Dense(layer_size, activation='relu'),
    Dense(layer_size, activation='relu'),
    Dense(3, activation='relu')
])

model.compile(optimizer='nadam', loss='mean_squared_error')

set_weights(model, load_weights_and_bises())

predicted_rgb = model.predict(X) * 255  # Rescale the output
predicted_rgb = predicted_rgb.reshape((height, width, 3)).astype(np.uint8)

# Create and save the generated image
generated_image = Image.fromarray(predicted_rgb)

rr.log("train_output", rr.Image(generated_image))