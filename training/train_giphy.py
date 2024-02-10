from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import rerun as rr
from fxpmath import Fxp
import tensorflow as tf

# List all available devices
devices = tf.config.list_physical_devices()
print(f"Devices: {devices}")

rr.init("train_image", spawn=True)

# Open the animated GIF
frames = []
with Image.open('giphy.gif') as img:
    keep_frame_if_zero = 0
    for frame in ImageSequence.Iterator(img):
        keep_frame_if_zero = (keep_frame_if_zero + 1) % 5
        if keep_frame_if_zero != 0:
            continue
        frame = frame.resize((64, 64))
        frame = frame.convert('RGB')
        frames.append(frame)

# Load the image to calculate the width and height
image = frames[0]
width, height = frames[0].size

rr.set_time_seconds("frame", 0)
rr.log("train_input_all", rr.Image(np.reshape(frames, ((width) * len(frames), (height), 3)).astype(np.uint8)))

# Log the input data
for i in range(len(frames)):
    rr.set_time_seconds("frame", i)
    rr.log("train_input", rr.Image(frames[i]))

# Prepare input data (X): Normalized pixel coordinates
x_coords = np.arange(width).reshape(-1, 1) / width  # Normalize x coordinates
y_coords = np.arange(height).reshape(-1, 1) / height  # Normalize y coordinates
frame_coords = np.linspace(0, 1, num=len(frames)).reshape(-1, 1) / len(frames)  # Normalize frame time

input_range = 64
def gen_input(x_coords, y_coords, frame_coords):
    # the sin functions introduce artifacts at the bottom
    return np.array([
        np.array([
            [
                np.sin(x * i),
                np.sin(y * i),
                np.cos(f * i)
            ] for i in range(input_range)
            ]).flatten()
        for y in y_coords for x in x_coords for f in frame_coords
        ])

X = gen_input(x_coords, y_coords, frame_coords)
rr.log("log", rr.TextLog(str(X.shape)))
#rr.log("input_vec", rr.Tensor(np.reshape(X, (256, 128, 128))))

# Prepare output data (Y): Normalized RGB values
pixels = np.array(frames) / 255.0  # Normalize pixel values
Y = pixels.reshape(-1, 3)  # Reshape to a 2D array where each row is RGB


print(f"X.shape : {X.shape}")
print(f"Y.shape : {Y.shape}")

layer_size = 64
model = Sequential([
    Dense(layer_size, input_dim=input_range * 3, activation='relu'),
    Dense(layer_size, activation='relu'),
    #Dense(layer_size, activation='relu'),
    Dense(3, activation='sigmoid')  # Use 'sigmoid' to output values between 0 and 1
])

model.compile(optimizer='nadam', loss='mean_squared_error')
history = model.fit(X, Y, epochs=20, batch_size=32, verbose=True, validation_split=.1)  # Adjust epochs and batch size as needed

# calculate distribution of weights of layers
w_dict = {}
for layer in model.layers:
    print(f"{layer.name}:\tshape: {layer.get_weights()[0].shape}")
    w_dict[layer.name] = model.get_layer(layer.name).get_weights()
    print('{} (weights):\tmean = {}\tstd = {}'.format(layer.name, np.mean(w_dict[layer.name][0]), np.std(w_dict[layer.name][0])))
    print('{} (bias):\t\tmean = {}\tstd = {}\n'.format(layer.name, np.mean(w_dict[layer.name][1]), np.std(w_dict[layer.name][1])))

def convert_to_fxp():
    # convert to fixed point
    fxp_ref = Fxp(None, dtype='fxp-s8/7')
    w_fxp_dict = {}
    for layer in w_dict.keys():
        w_fxp_dict[layer] = [
            Fxp(w_dict[layer][0], like=fxp_ref), 
            Fxp(w_dict[layer][1], like=fxp_ref),
            ]
        
    # update model with fixed point and evaluate
    for layer, values in w_fxp_dict.items():
        model.get_layer(layer).set_weights(values)

    # write out layers of fixed point weights as a binary with a file per layer
    for layer, values in w_fxp_dict.items():
        i = 0
        values, biases = values
        for v_array in values:
            #print(repr(v_array))
            int_values = np.array([int(val) for val in v_array], dtype=np.int8)
            with open(f"output/{layer}_weights_{i}.bin", "wb") as f:
                int_values.tofile(f)
            i += 1

convert_to_fxp()

num_output_frames = len(frames) * 2
#x_coords = np.linspace(-64, 128, num=64+128).reshape(-1, 1) / width  # Normalize x coordinates
#y_coords = np.linspace(-64, 128, num=64+128).reshape(-1, 1) / height  # Normalize y coordinates
frame_coords = np.linspace(0, 1, num=num_output_frames).reshape(-1, 1) / len(frames)  # Normalize frame time

X = gen_input(x_coords, y_coords, frame_coords)

predicted_rgb = model.predict(X) * 255  # Rescale the output

rr.log("log", rr.TextLog(str(predicted_rgb.shape)))

predicted_rgb = predicted_rgb.reshape((len(x_coords) * num_output_frames, len(y_coords), 3)).astype(np.uint8)

# Create and save the generated image
rr.set_time_seconds("frame", 0)
rr.log("train_output_all", rr.Image(predicted_rgb))

for i in range(num_output_frames):
    time = i / num_output_frames * len(frames)
    rr.set_time_seconds("frame", time)
    rr.log("train_output", rr.Image(predicted_rgb[i*len(x_coords):(i+1)*len(x_coords), :, :]))
