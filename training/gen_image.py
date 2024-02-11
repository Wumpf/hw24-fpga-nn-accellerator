import os
import rerun as rr
import struct

import numpy as np

from PIL import Image

from common import NeuralNetwork, quantize_array, write_array_to_memh_file

def quantize_array_2(values):
    d = 255.0/8.0
    values = np.clip(values, 0.0, 8.0)
    values *= d
    values = np.floor(values)
    values /= d
    return values

def write_array_to_memh_file_2(values, name, folder):
    with open(os.path.join(folder, f"{name}.txt"), "w") as f:
        for m in values.flatten() * (255.0/8.0):
            m = int(m)
            m_signed_byte = struct.pack('B', m)
            assert m_signed_byte >= struct.pack('B', 0)
            f.write(f"{m_signed_byte.hex()} ")

rr.init("test_ascii", spawn=True)

if True:
    mode = 'RGBA'
    channels = 4
else:
    mode = 'RGB'
    channels = 3

width, height = 64, 64
embedding_size = 32
layer_size = 64

nn = NeuralNetwork(width, height, layer_size, embedding_size, channels)

nn.load_weights_from_folder("output_with_eyes")
predicted_rgb = nn.predict()
generated_image = Image.fromarray(predicted_rgb, 'RGBA')

rr.log("train_output", rr.Image(generated_image))

activations = nn.predict_with_activations()

write_array_to_memh_file(quantize_array(nn.encoded_pos()), "encoded_pos", "output_with_eyes")

for i, a in enumerate(activations):
    qa = quantize_array_2(a)
    print(f"min/max activation_{i}: {np.min(a)} / {np.max(a)}, mean: {np.mean(a)}, std: {np.std(a)}")
    print(f"min/max quantized activation_{i}: {np.min(qa)} / {np.max(qa)}, mean: {np.mean(qa)}, std: {np.std(qa)}")
    write_array_to_memh_file_2(
        quantize_array_2(a), f"activation_{i}", "output_with_eyes")
    
for activation_index, a in enumerate(activations[:-1]):
    size = 64
    if (activation_index == len(activations) - 2):
        size = 32
    array = a.reshape((64, 64, size))

    normalized_array = 255 * (array - np.min(array)) / (np.max(array) - np.min(array))
    normalized_array = normalized_array.astype(np.uint8)

    # Create a blank 4096x4096 image
    image = Image.new('L', (512, 512))

    # Place each 64x64 slice in the 4096x4096 image
    for i in range(size):  # For each slice        
        # Extract the 64x64 slice
        slice = normalized_array[:, :, i].reshape(64, 64)

        # Convert slice to PIL image
        slice_image = Image.fromarray(slice, 'L')
        x = i % 8 * 64
        y = i // 8 * 64
        print(f"i: {i}, x: {x}, y: {y}")

        # Calculate the position
        position = (x, y)

        # Paste the slice image into the main image
        image.paste(slice_image, position)

    rr.log(f"activation_{activation_index}", rr.Image(image))

rr.log("last_activation", rr.Image((activations[-1] * 255).reshape((64, 64, 4)).astype(np.uint8)))
rr.log("last_activation_quantized", rr.Image((quantize_array_2(activations[-1]) * 255).reshape((64, 64, 4)).astype(np.uint8)))
