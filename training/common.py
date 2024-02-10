import os
import shutil
import struct
import numpy as np

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

class NeuralNetwork(object):
    def __init__(self, img_width, img_height, layer_size, embedding_size, channels) -> None:
        self.__img_width = img_width
        self.__img_height = img_height
        self.__layer_size = layer_size
        self.__embedding_size = embedding_size
        self.__channels = channels

        self.__x_coords = np.arange(self.__img_width).reshape(-1, 1) / self.__img_width
        self.__y_coords = np.arange(self.__img_height).reshape(-1, 1) / self.__img_height

        self.__encoded_pos = self.__encode_positions_interleaved_bits()
        #self.__encoded_pos = self.__encode_positions()

        self.__model = Sequential([
            Dense(self.__layer_size, input_dim=embedding_size*2, activation='relu'),
            Dense(self.__layer_size, activation='relu'),
            Dense(self.__layer_size, activation='relu'),
            Dense(self.__channels, activation='hard_sigmoid')
        ])
        self.__model.compile(optimizer='nadam', loss='mean_squared_error')
    
    def __encode_positions_interleaved_bits(self):
        return np.array([self.__interleave_bits(x, y) for x in self.__x_coords for y in self.__y_coords])

    def __float_to_binary(self, float_value):
        return format(struct.unpack('!I', struct.pack('!f', float_value))[0], '032b')

    def __interleave_bits(self, x, y):
        bin_x = self.__float_to_binary(x)
        bin_y = self.__float_to_binary(y * 6057489)
        
        return np.array([int(bit) for pair in zip(bin_x, bin_y) for bit in pair])
    
    def __encode_positions(self):
        seed = 127
        np.random.seed(seed)
        rnd = np.random.random_integers(0, 0xFFFFFF, self.__embedding_size)
        return np.array([
            np.array([
                [np.sin(x * rnd[i] + y), np.cos(y * rnd[self.__embedding_size-1-i] + x)]
                    for i in range(self.__embedding_size)]).flatten()
                        for x in self.__x_coords for y in self.__y_coords])

    def train(self, image, epochs=100, batch_size=32):
        Y = (np.array(image) / 255.0).reshape(-1, self.__channels)
        return self.__model.fit(self.__encoded_pos, Y, epochs=epochs, batch_size=batch_size, verbose=True, validation_split=.1)  # Adjust epochs and batch size as needed

    def __load_ascii(self, folder, name):
        with open(f"{folder}/{name}.txt", "r") as f:
            floats = np.array([float(int(x, 16)) for x in f.read().strip().split(" ")])
            floats -= 127
            floats /= 128.0
            return floats
        
    def __load_ascii_reshape(self, folder, name, x, y):
        with open(f"{folder}/{name}.txt", "r") as f:
            floats = np.array([float(int(x, 16)) for x in f.read().strip().split(" ")])
            floats -= 127
            floats /= 128.0
            return floats.reshape((x, y))

    def load_weights_from_folder(self, folder):
        layers = {
                "dense": (self.__load_ascii_reshape(folder, "dense_weights", 64, 64), self.__load_ascii(folder, "dense_biases")),
                "dense_1": (self.__load_ascii_reshape(folder, "dense_1_weights", 64, 64), self.__load_ascii(folder, "dense_1_biases")),
                "dense_2": (self.__load_ascii_reshape(folder, "dense_2_weights", 64, 64), self.__load_ascii(folder, "dense_2_biases")),
                "dense_3": (self.__load_ascii_reshape(folder, "dense_3_weights", 64, self.__channels), self.__load_ascii(folder, "dense_3_biases"))
            }
        for layer, values in layers.items():
                self.__model.get_layer(layer).set_weights(values)

    def predict(self):
        predicted_rgba = self.__model.predict(self.__encoded_pos) * 255  # Rescale the output
        return predicted_rgba.reshape((self.__img_height, self.__img_width, self.__channels)).astype(np.uint8)

    def __quantize_parameters(self):
        w_dict = {}
        for layer in self.__model.layers:
            w_dict[layer.name] = self.__model.get_layer(layer.name).get_weights()

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

    def __write_array_to_memh_file(self, values, name, folder):
        with open(os.path.join(folder, f"{name}.txt"), "w") as f:
            for m in values.flatten() * 128.0 + 127.0:
                f.write(f"{int(m):0{2}x} ")

    def __write_weights(self, quantized_params, folder):
        """Write the weights to binary files."""
        if os.path.exists(folder):
            shutil.rmtree(folder)
            
        os.makedirs(folder, exist_ok=True)
        for layer, values in quantized_params.items():
            self.__write_array_to_memh_file(values[0], f"{layer}_weights", folder)
            self.__write_array_to_memh_file(values[1], f"{layer}_biases", folder)
    
    def write_quantized_weights(self, folder):
        quantized_params = self.__quantize_parameters()
        for layer, values in quantized_params.items():
            self.__model.get_layer(layer).set_weights(values)
        self.__write_weights(quantized_params, folder)