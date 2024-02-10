import os
import shutil
import rerun as rr

import numpy as np
from fxpmath import Fxp
import tensorflow as tf

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense

def load_weights_and_bises(folder):
    lengths = {
        "dense_weights": (64, 64),
        "dense_biases": (64,),
        "dense_1_weights": (64, 64),
        "dense_1_biases": (64,),
        "dense_2_weights": (64, 64),
        "dense_2_biases": (64,),
        "dense_3_weights": (64, 3),
        "dense_3_biases": (3,)
    }

    def convert_byte_to_sbyte(byte):
        if byte > 127:
            return byte - 256
        return byte

    def byte_to_fixed_point(byte_value):
        """
        Convert a signed byte (int8) to a signed fixed-point value with 7 fractional bits.

        Parameters:
        - byte_value: An integer representing the signed byte value. Should be in range [-128, 127].

        Returns:
        - A float representing the fixed-point value.
        """
        # Check if input is within the valid range for int8
        if byte_value < -128 or byte_value > 127:
            raise ValueError("Value must be in the range of a signed byte: [-128, 127]")

        # Since there are 7 bits for the fractional part, divide by 2**7 to convert
        fixed_point_value = byte_value / 128.0

        return fixed_point_value

    def load_fxp_array_from_binary_file(path):
        def byte_to_fxp(byte):
            print(f"byte: {byte} = > {byte_to_fixed_point(byte)}")
            return byte_to_fixed_point(byte)
        with open(path, 'rb') as file:
            bytes_data = file.read()
            return [byte_to_fxp(convert_byte_to_sbyte(byte)) for byte in bytes_data]

    w_fxp_dict = {}
    for file in os.listdir(folder):
        print(f"FILE: {file}")
        if file.endswith("_weights.bin"):
            layer = file[:-12]
            weights = np.array(
                load_fxp_array_from_binary_file(os.path.join(folder, file)), dtype=np.float32)
            weights = weights.reshape(lengths[f"{layer}_weights"][0], lengths[f"{layer}_weights"][1])
            if layer not in w_fxp_dict.keys():
                w_fxp_dict[layer] = (None, None)
            w_fxp_dict[layer] = (weights, w_fxp_dict[layer][1])
        elif file.endswith("_biases.bin"):
            layer = file[:-11]
            biases = np.array(
                load_fxp_array_from_binary_file(os.path.join(folder, file)), dtype=np.float32)
            if layer not in w_fxp_dict.keys():
                w_fxp_dict[layer] = (None, None)
            w_fxp_dict[layer] = (w_fxp_dict[layer][0], biases)
    
    return w_fxp_dict

def set_weights(model, w_fxp_dict):
    # update model with fixed point and evaluate
    for layer, values in w_fxp_dict.items():
        model.get_layer(layer).set_weights(values)

# List all available devices
devices = tf.config.list_physical_devices()
print(f"Devices: {devices}")

rr.init("test_binary", spawn=True)

width, height = 64, 64

# Prepare input data (X): Normalized pixel coordinates
x_coords = np.arange(width).reshape(-1, 1) / width
y_coords = np.arange(height).reshape(-1, 1) / height

def x_training_data(x_coords, y_coords, embedding_size):
    return np.array([np.array([[np.sin(x * i), np.sin(y * i)] for i in range(embedding_size)]).flatten() for y in y_coords for x in x_coords])

def x_training_data_rnd(x_coords, y_coords, embedding_size):
    rnd = np.random.random_integers(0, 0xFFFFFF, embedding_size)
    return np.array([np.array([[np.sin(x * rnd[i]), np.sin(y * rnd[embedding_size-1-i])] for i in range(embedding_size)]).flatten() for y in y_coords for x in x_coords])

def x_training_data_rnd_sin_cos(x_coords, y_coords, embedding_size):
    rnd = np.random.random_integers(0, 0xFFFFFF, embedding_size)
    return np.array([np.array([[np.sin(x * rnd[i]), np.cos(y * rnd[embedding_size-1-i])] for i in range(embedding_size)]).flatten() for y in y_coords for x in x_coords])

def x_training_data_rnd_sin_cos_biased(x_coords, y_coords, embedding_size):
    rnd = np.random.random_integers(0, 0xFFFFFF, embedding_size)
    return np.array([np.array([[np.sin(x * rnd[i] + y), np.cos(y * rnd[embedding_size-1-i] + x)] for i in range(embedding_size)]).flatten() for y in y_coords for x in x_coords])

embedding_size = 32
layer_size = 64

# X = x_training_data_rnd(x_coords, y_coords, embedding_size)
# X = x_training_data_rnd_sin_cos(x_coords, y_coords, embedding_size)
X = x_training_data_rnd_sin_cos_biased(x_coords, y_coords, embedding_size)

rr.log("log", rr.TextLog(str(X.shape)))

print(f"X.shape : {X.shape}")

model = Sequential([
    Dense(layer_size, input_dim=embedding_size*2, activation='relu'),
    Dense(layer_size, activation='relu'),
    Dense(layer_size, activation='relu'),
    Dense(3, activation='sigmoid')  # Use 'sigmoid' to output values between 0 and 1
])

model.compile(optimizer='nadam', loss='mean_squared_error')

w_fxp_dict = load_weights_and_bises("output")
print(w_fxp_dict)
set_weights(model, w_fxp_dict)

predicted_rgb = model.predict(X) * 255  # Rescale the output
predicted_rgb = predicted_rgb.reshape((height, width, 3)).astype(np.uint8)

# Create and save the generated image
generated_image = Image.fromarray(predicted_rgb)

rr.log("train_output", rr.Image(generated_image))