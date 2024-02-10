import rerun as rr

import numpy as np

from PIL import Image

from common import NeuralNetwork

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

rr.init("test_ascii", spawn=True)

width, height = 64, 64
embedding_size = 32
layer_size = 64

nn = NeuralNetwork(width, height, layer_size, embedding_size)
nn.load_weights_from_folder("output")
predicted_rgb = nn.predict()
generated_image = Image.fromarray(predicted_rgb)

rr.log("train_output", rr.Image(generated_image))