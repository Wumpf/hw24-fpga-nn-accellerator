import os
import shutil
import rerun as rr

import numpy as np
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

def float_array_to_fxp(array):
    return [float_to_fxp(val) for val in array]

def float_to_fxp(value):
    return 0

def quantize_parameters(model):
    w_dict = {}
    for layer in model.layers:
        w_dict[layer.name] = model.get_layer(layer.name).get_weights()

    quantized_params = {}
    for layer in w_dict.keys():
        weights, biases = w_dict[layer]

        d = 128.0

        weights = np.clip(weights, -1.0, 1.0)
        weights *= d
        weights = np.floor(weights)
        weights /= d

        biases = np.clip(biases, -1.0, 1.0)
        biases *= d
        biases = np.floor(biases)
        biases /= d

        quantized_params[layer] = (weights, biases)
    
    return quantized_params

def set_weights(model, quantized_params):
    # update model with fixed point and evaluate
    for layer, values in quantized_params.items():
        model.get_layer(layer).set_weights(values)

def write_array_to_file(values, name, folder):
    with open(os.path.join(folder, f"{name}.bin"), "wb") as f:
        values.flatten().tofile(f)

def write_array_to_memh_file(values, name, folder):
    with open(os.path.join(folder, f"{name}.txt"), "w") as f:
        for m in values.flatten() * 128.0 + 127.0:
            f.write(f"{int(m):0{2}x} ")

def write_weights(quantized_params):
    """Write the weights to binary files."""
    if os.path.exists("output"):
        shutil.rmtree("output")
        
    os.makedirs("output", exist_ok=True)
    for layer, values in quantized_params.items():
        write_array_to_memh_file(values[0], f"{layer}_weights", "output")
        write_array_to_memh_file(values[1], f"{layer}_biases", "output")

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

def x_training_data_rnd_sin_cos_biased(x_coords, y_coords, embedding_size):
    seed = 127
    np.random.seed(seed)
    rnd = np.random.random_integers(0, 0xFFFFFF, embedding_size)
    return np.array([np.array([[np.sin(x * rnd[i] + y), np.cos(y * rnd[embedding_size-1-i] + x)] for i in range(embedding_size)]).flatten() for y in y_coords for x in x_coords])

embedding_size = 32
layer_size = 64

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

quantized_params = quantize_parameters(model)
set_weights(model, quantized_params)
write_weights(quantized_params)

predicted_rgb = model.predict(X) * 255  # Rescale the output
predicted_rgb = predicted_rgb.reshape((height, width, 3)).astype(np.uint8)

# Create and save the generated image
generated_image = Image.fromarray(predicted_rgb)

rr.log("train_output", rr.Image(generated_image))