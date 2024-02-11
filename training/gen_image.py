import rerun as rr

import numpy as np

from PIL import Image

from common import NeuralNetwork, write_array_to_memh_file, quantize_array

def quantize_array_2(values):
    d = 256.0/8.0
    values = np.clip(values, 0.0, 8.0)
    values *= d
    values = np.floor(values)
    values /= d
    return values

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

write_array_to_memh_file(quantize_array_2(nn.encoded_pos()), "encoded_pos", "output_with_eyes")

for i, a in enumerate(activations):
    qa = quantize_array_2(a)
    print(f"min/max activation_{i}: {np.min(a)} / {np.max(a)}, mean: {np.mean(a)}, std: {np.std(a)}")
    print(f"min/max quantized activation_{i}: {np.min(qa)} / {np.max(qa)}, mean: {np.mean(qa)}, std: {np.std(qa)}")
    write_array_to_memh_file(
        quantize_array_2(a), f"activation_{i}", "output_with_eyes")

rr.log("last_activation", rr.Image((activations[-1] * 255).reshape((64, 64, 4)).astype(np.uint8)))
rr.log("last_activation_quantized", rr.Image((quantize_array_2(activations[-1]) * 255).reshape((64, 64, 4)).astype(np.uint8)))
