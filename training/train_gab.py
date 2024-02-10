import os
import shutil
import rerun as rr

import numpy as np
from fxpmath import Fxp
import tensorflow as tf

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense

def gen_gradient_image(w, h):
    """Generate a gradient image of size w x h."""
    image = np.zeros((w, h, 3), dtype=np.uint8)
    for y in range(w):
        image[:, y, 0] = np.linspace(0, 255, w)
    for x in range(h):
        image[x, :, 1] = np.linspace(0, 255, h)
    return image

def convert_to_fxp(model):
    w_dict = {}
    for layer in model.layers:
        w_dict[layer.name] = model.get_layer(layer.name).get_weights()

    # convert to fixed point
    fxp_ref = Fxp(None, dtype='fxp-s8/7')
    w_fxp_dict = {}
    for layer in w_dict.keys():
        w_fxp_dict[layer] = (
            Fxp(w_dict[layer][0], like=fxp_ref),
            Fxp(w_dict[layer][1], like=fxp_ref)
        )
    
    return w_fxp_dict

def set_weights(model, w_fxp_dict):
    # update model with fixed point and evaluate
    for layer, values in w_fxp_dict.items():
        model.get_layer(layer).set_weights(values)

def write_fpx_weights_to_file(values, name, folder):
    print(f"write_fpx_weights_to_file {name} to {folder}, {len(values)} values")
    int_values = np.array([], dtype=np.int8)
    for v_array in values:
        int_values = np.append(int_values, np.array([val.bin() for val in v_array], dtype=np.int8))
    with open(os.path.join(folder, f"{name}.bin"), "wb") as f:
        int_values.flatten().tofile(f)

def write_fpx_biases_to_file(values, name, folder):
    print(f"write_fpx_biases_to_file {name} to {folder}, {len(values)} values")
    print(f"values: {values}")
    int_values = np.array([val.bin() for val in values], dtype=np.int8)
    with open(os.path.join(folder, f"{name}.bin"), "wb") as f:
        print(f"output {name} => {len(int_values)} values")
        int_values.flatten().tofile(f)

def write_weights(w_fxp_dict):
    """Write the weights to binary files."""
    if os.path.exists("output"):
        shutil.rmtree("output")
        
    os.makedirs("output", exist_ok=True)
    for layer, values in w_fxp_dict.items():
        write_fpx_weights_to_file(values[0], f"{layer}_weights", "output")
        write_fpx_biases_to_file(values[1], f"{layer}_biases", "output")

# List all available devices
devices = tf.config.list_physical_devices()
print(f"Devices: {devices}")

rr.init("train_image", spawn=True)

# Load the image
image_path = 'pretty_256.png'
image = Image.open(image_path)
image = image.resize((64, 64))
image = image.convert('RGB')
width, height = image.size

rr.log("train_input", rr.Image(image))

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

# Prepare output data (Y): Normalized RGB values
  # Reshape to a 2D array where each row is RGB
Y = (np.array(image) / 255.0).reshape(-1, 3)

print(f"X.shape : {X.shape}")
print(f"Y.shape : {Y.shape}")

model = Sequential([
    Dense(layer_size, input_dim=embedding_size*2, activation='relu'),
    Dense(layer_size, activation='relu'),
    Dense(layer_size, activation='relu'),
    Dense(3, activation='sigmoid')  # Use 'sigmoid' to output values between 0 and 1
])

model.compile(optimizer='nadam', loss='mean_squared_error')
history = model.fit(X, Y, epochs=100, batch_size=32, verbose=True, validation_split=.1)  # Adjust epochs and batch size as needed

w_fxp_dict = convert_to_fxp(model)
set_weights(model, w_fxp_dict)
write_weights(w_fxp_dict)

predicted_rgb = model.predict(X) * 255  # Rescale the output
predicted_rgb = predicted_rgb.reshape((height, width, 3)).astype(np.uint8)

# Create and save the generated image
generated_image = Image.fromarray(predicted_rgb)

rr.log("train_output", rr.Image(generated_image))