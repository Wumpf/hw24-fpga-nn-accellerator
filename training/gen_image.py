import rerun as rr

import numpy as np

from PIL import Image

from common import NeuralNetwork, write_array_to_memh_file, quantize_array

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

nn.load_weights_from_folder("output/with_eyes")
predicted_rgb = nn.predict()
generated_image = Image.fromarray(predicted_rgb, 'RGBA')

rr.log("train_output", rr.Image(generated_image))

activations = nn.predict_with_activations()

for i, a in enumerate(activations):
    write_array_to_memh_file(quantize_array(a), f"activation_{i}", "output/with_eyes")
