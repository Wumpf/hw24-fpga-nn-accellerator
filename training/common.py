import os
import shutil
import struct
import numpy as np

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, LayerNormalization, UnitNormalization, Lambda, ReLU, Activation
from keras.optimizers import Adam, Nadam


def gen_gradient_image(w, h):
    """Generate a gradient image of size w x h."""
    image = np.zeros((w, h, 3), dtype=np.uint8)
    for y in range(w):
        image[:, y, 0] = np.linspace(0, 255, w)
    for x in range(h):
        image[x, :, 1] = np.linspace(0, 255, h)
    return image

def quantize_array(values):
    d = 127.0
    values = np.clip(values, -1.0, 1.0)
    values *= d
    values = np.floor(values)
    values /= d
    return values

def write_array_to_memh_file(values, name, folder):
    with open(os.path.join(folder, f"{name}.txt"), "w") as f:
        for m in values.flatten() * 127.0:
            m = int(m)
            assert -127 <= m <= 127
            m_signed_byte = struct.pack('b', m)
            f.write(f"{m_signed_byte.hex()} ")

class NeuralNetwork(object):
    def __init__(self, img_width, img_height, layer_size, embedding_size, channels, learning_rate=1e-3, mixed_precision=None, normalization_type=None, activation_quantizer=None) -> None:

        if mixed_precision != None:
            # supported types: mixed_float16, mixed_bfloat16
            tf.keras.mixed_precision.set_global_policy(mixed_precision)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        # see also: https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer
        # see also: https://www.tensorflow.org/api_docs/python/tf/keras/constraints/MinMaxNorm

        self.__img_width = img_width
        self.__img_height = img_height
        self.__layer_size0 = layer_size
        self.__layer_size1 = layer_size
        self.__layer_size2 = layer_size // 2
        self.__embedding_size = embedding_size
        self.__channels = channels

        self.__x_coords = np.arange(self.__img_width).reshape(-1, 1) / self.__img_width
        self.__y_coords = np.arange(self.__img_height).reshape(-1, 1) / self.__img_height

        mx = 654
        my = 57436
        # self.__encoded_pos = self.__encode_positions_bits(mx, my)
        # self.__encoded_pos = self.__encode_positions_bitswap()
        # self.__encoded_pos = self.__encode_positions_sincos()
        # self.__encoded_pos = self.__encode_positions_sincos2()
        # self.__encoded_pos = self.__encode_positions_tri()
        # self.__encoded_pos, self.__encoding_matrix = self.__encode_positions_sawtooth()
        self.__encoded_pos, self.__encoding_matrix = self.__encode_positions_sawtooth_fixed()

        # print (self.__encoded_pos.shape)
        # self.__encoded_pos = np.array([[x[0],y[0]] for x in self.__x_coords for y in self.__y_coords])#.reshape((64, 2))
        # print (self.__encoded_pos.shape)

        print (f'envoded pos min: {np.min(self.__encoded_pos)}, max: {np.max(self.__encoded_pos)}, mean: {np.mean(self.__encoded_pos)}, std: {np.std(self.__encoded_pos)}')
        # print (self.__encoded_pos[0:64])
        # print (self.__encoded_pos[0:128]*127)

        normalization = LayerNormalization if normalization_type == "Norm" else None
        normalization = LayerNormalization if normalization_type == "LayerNorm" else normalization
        normalization = BatchNormalization if normalization_type == "BatchNorm" else normalization
        normalization = normalization_type if normalization_type != type(str) and normalization == None else normalization

        activation = 'relu6'
        last_activation = 'sigmoid'
        # last_activation = 'hard_sigmoid'
        # last_activation = 'relu6'

        # hard sigmoid formula
        # if x < -2.5: return 0
        # if x > 2.5: return 1
        # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5

        if activation_quantizer != None:
            activation_with_quantization = lambda x: tf.keras.activations.relu(activation_quantizer(x))
        else:
            activation_with_quantization = activation
        pre_activation_quantization = activation_quantizer

        use_bias = False

        # seed = 127
        # np.random.seed(seed)
        # self.__rnd = np.random.random_integers(0, 127, self.__embedding_size*4)
        # assert np.all(self.__encoding_matrix == self.__rnd)
        # print (self.__rnd)

        if normalization == None:
            self.__model = Sequential([
                # Dense(self.__embedding_size*2, use_bias=use_bias, input_dim=2, kernel_initializer=initializers.Zeros()), # initializers.RandomNormal(mean=.5, stddev=.5*.5)),
                # Activation(activation=lambda x: tf.sin(x*3.14)),
                # Dense(self.__layer_size0, use_bias=use_bias, activation=activation_with_quantization),

                Dense(self.__layer_size0, use_bias=use_bias, activation=activation_with_quantization, input_dim=embedding_size*2),
                Dense(self.__layer_size1, use_bias=use_bias, activation=activation_with_quantization),
                Dense(self.__layer_size2, use_bias=use_bias, activation=activation_with_quantization),
                Dense(self.__channels,    use_bias=use_bias, activation=pre_activation_quantization),
                Activation(activation=last_activation)
            ])
        else:
            self.__model = Sequential([
                Dense(self.__layer_size0, use_bias=use_bias, activation=activation_with_quantization, input_dim=embedding_size*2),
                normalization(),
                Dense(self.__layer_size1, use_bias=use_bias, activation=activation_with_quantization),
                normalization(),
                Dense(self.__layer_size2, use_bias=use_bias, activation=activation_with_quantization),
                normalization(),
                Dense(self.__channels,    use_bias=use_bias, activation=pre_activation_quantization),
                Activation(activation=last_activation)
            ])
        # layer = self.__model.get_layer("dense").set_weights([(self.__rnd.astype(float)).reshape((2,64))])

        self.__model.compile(optimizer=Nadam(learning_rate=learning_rate), loss='mean_squared_error')
        print(self.__model.optimizer.learning_rate.numpy())
    
    def encoded_pos(self):
        return self.__encoded_pos
    
    def __encode_positions_bits(self, mx, my):
        return np.array([self.__interleave_bits(x, y, mx, my) for x in self.__x_coords for y in self.__y_coords])

    def __float_to_binary(self, float_value):
        return format(struct.unpack('!I', struct.pack('!f', float_value))[0], '032b')

    def __interleave_bits(self, x, y, mx, my):
        bin_x = self.__float_to_binary(x * mx)
        bin_y = self.__float_to_binary(y * my)
        
        return np.array([int(bit) for pair in (bin_x + bin_y) for bit in pair])
    
    def __encode_positions(self, periodic_fnA=np.sin, periodic_fnB=np.cos, random_range=(0, 0xFFFFFF), entangled=True):
        seed = 127
        np.random.seed(seed)
        rnd = np.random.random_integers(random_range[0], random_range[1], self.__embedding_size)
        return np.array([
            np.array([
                [periodic_fnA(x * rnd[i] + y), periodic_fnB(y * rnd[self.__embedding_size-1-i] + x)] if entangled else
                [periodic_fnA(x * rnd[i]), periodic_fnB(y * rnd[self.__embedding_size-1-i])]
                    for i in range(self.__embedding_size)]).flatten()
                        for x in self.__x_coords for y in self.__y_coords])

    def __encode_positions_sincos2(self, periodic_fnA=np.sin, periodic_fnB=np.cos, random_range=(0, 0xFFFFFF), entangled=True):
        seed = 127
        np.random.seed(seed)
        rnd = np.random.random_integers(random_range[0], random_range[1], self.__embedding_size*4)
        return np.array([
            np.array([
                [periodic_fnA(x * rnd[i*4+0] + y * rnd[i*4+1]),
                 periodic_fnB(y * rnd[i*4+2] + x * rnd[i*4+3])]
                    for i in range(self.__embedding_size)]).flatten()
                        for x in self.__x_coords for y in self.__y_coords])

    def __encode_positions_sincos(self, entangled=True):
        return self.__encode_positions(np.sin, np.cos, entangled=entangled)

    def __encode_positions_sawtooth(self, entangled=True):
        periodic_fn = lambda x : np.modf(x)[0]*2-1 if int(np.modf(x)[1]) % 1 == 0 else (1-np.modf(x)[0])*2-1

        seed = 127
        np.random.seed(seed)
        rnd = np.random.random_integers(0, 127, self.__embedding_size*4)
        return np.array([
            np.array([
                [periodic_fn(x * rnd[i*4+0] + y * rnd[i*4+1]),
                 periodic_fn(x * rnd[i*4+2] + y * rnd[i*4+3])]
                    for i in range(self.__embedding_size)]).flatten()
                        for x in self.__x_coords for y in self.__y_coords]), rnd

    def __encode_positions_sawtooth_fixed(self):
        def toSigned8(n):
            n = n & 0xff
            return n | (-(n & 0x80))

        periodic_fn = lambda x : toSigned8(x&0xFF if (x&0x100) == 0 else 0xFF-(x&0xFF)) / 127.0

        seed = 127
        np.random.seed(seed)
        rnd = np.random.random_integers(0, 127, self.__embedding_size*4)
        return np.array([
            [periodic_fn(int(x*128) * rnd[i*2+0] + int(y*128) * rnd[i*2+1]) for i in range(self.__embedding_size*2)]
                for x in self.__x_coords for y in self.__y_coords]), rnd


    def __encode_positions_tri(self, entangled=True):
        periodic_fn = lambda x : np.modf(x)[0]*2-1
        return self.__encode_positions(periodic_fn, periodic_fn, entangled=entangled)

    def __encode_positions_bitswap(self, random_range=(0, 255)):
        def bitswap(x, y, rnd):
            # print (x, y, rnd)
            # src_bits = \
            #     format(struct.unpack('!B', struct.pack('b',    x+y))[0], '06b') + \
            #     format(struct.unpack('!B', struct.pack('b',    x-y))[0], '06b') + \
            #     format(struct.unpack('!B', struct.pack('b', 64-x+y))[0], '06b') + \
            #     format(struct.unpack('!B', struct.pack('b', 64-x-y))[0], '06b')
            src_bits = \
                format(struct.unpack('!B', struct.pack('b',    x))[0], '06b') + \
                format(struct.unpack('!B', struct.pack('b',    y))[0], '06b') + \
                format(struct.unpack('!B', struct.pack('b', 64-x))[0], '06b') + \
                format(struct.unpack('!B', struct.pack('b', 64-y))[0], '06b')
            bits = [src_bits[i%len(src_bits)] for i in rnd]
            # print (''.join(src_bits), ''.join(bits))
            # print (bits, ''.join(bits))
            return int(''.join(bits), 2)/128-1.0

        # print(self.__x_coords.shape)
        # print(self.__x_coords)

        seed = 127
        np.random.seed(seed)
        rnd = np.random.random_integers(random_range[0], random_range[1], self.__embedding_size * 16)
        return np.array([
            np.array([
                # [bitswap(int(x*64), int(y*64), rnd[(i+0)*8 : (i+1)*8]),
                #  bitswap(int(x*64), int(y*64), rnd[(i+2)*8 : (i+3)*8])]
                [bitswap(int(x*64), int(y*64), np.cumsum([rnd[i]]*8)),
                 bitswap(int(y*64), int(x*64), np.cumsum([rnd[self.__embedding_size-1-i]]* 8))]
                    for i in range(self.__embedding_size)]).flatten()
                        for x in self.__x_coords.flatten() for y in self.__y_coords.flatten()])

    def train(self, image, epochs=100, batch_size=32):
        Y = (np.array(image) / 255.0).reshape(-1, self.__channels)
        return self.__model.fit(self.__encoded_pos, Y, epochs=epochs, batch_size=batch_size, verbose=True)  # Adjust epochs and batch size as needed

    # rr.set_time_sequence("step", step)
    # rr.log("scalar", rr.Scalar(math.sin(step / 10.0)))

    def __load_ascii(self, folder, name, verbose=False):
        with open(f"{folder}/{name}.txt", "r") as f:
            hex_values = f.read().strip().split(" ")
            floats = np.array(
                [int.from_bytes(
                    int(value, 16).to_bytes(1, byteorder='little', signed=False), 'little', signed=True) for value in hex_values], dtype=np.float32)
            if verbose:
                print(floats)
            floats /= 128.0
            if verbose:
                print(floats)
            return floats
        
    def __load_ascii_reshape(self, folder, name, x, y):
        return self.__load_ascii(folder, name).reshape((x, y))

    def load_weights_from_folder(self, folder):
        layers = {
            "dense": (
                self.__load_ascii_reshape(folder, "dense_weights", 64, self.__layer_size0),
                self.__load_ascii(folder, "dense_biases"),
            ),
            "dense_1": (
                self.__load_ascii_reshape(folder, "dense_1_weights", self.__layer_size0, self.__layer_size1),
                self.__load_ascii(folder, "dense_1_biases"),
            ),
            "dense_2": (
                self.__load_ascii_reshape(folder, "dense_2_weights", self.__layer_size1, self.__layer_size2),
                self.__load_ascii(folder, "dense_2_biases"),
            ),
            "dense_3": (
                self.__load_ascii_reshape(folder, "dense_3_weights", self.__layer_size2, self.__channels),
                self.__load_ascii(folder, "dense_3_biases"),
            ),
        }
        for layer_name, values in layers.items():
            # set only as many parameter tensors as layer requires, skips bias in Dense layer that were trained without them
            layer = self.__model.get_layer(layer_name)
            layer.set_weights(values[:len(layer.get_weights())])

    def predict(self):
        predicted_rgba = self.__model.predict(self.__encoded_pos) * 255  # Rescale the output
        return predicted_rgba.reshape(
            (self.__img_height, self.__img_width, self.__channels)).astype(np.uint8)
    
    def predict_with_activations(self):
        layer_outputs = [layer.output for layer in self.__model.layers]  # Add all layers' output you're interested in here
        activation_model = Model(inputs=self.__model.input, outputs=layer_outputs)
        return activation_model.predict(self.__encoded_pos)

    def __quantize_parameters(self):
        w_dict = {}
        for layer in self.__model.layers:
            weights = self.__model.get_layer(layer.name).get_weights()
            if len(weights) == 1:
                w_dict[layer.name] = [weights[0], np.zeros(weights[0].shape[1])] # force biases to 0, if trained without them
            elif len(weights) > 1:
                w_dict[layer.name] = weights
            # print(f"name: {layer.name}, weights.shape: {w_dict[layer.name][0].shape}, biases.shape: {w_dict[layer.name][1].shape}")
            # print(f"first 2 elements ( {layer.name}.weights): {w_dict[layer.name][0][:2]}")
            # print(f"first flattened 2 elements ( {layer.name}.weights): {w_dict[layer.name][0].flatten()[:2]}")

        quantized_params = {}
        for layer in w_dict.keys():
            weights, biases = w_dict[layer]

            quantized_params[layer] = (
                quantize_array(weights),
                quantize_array(biases)
            )
        
        return quantized_params

    def __write_weights(self, quantized_params, folder):
        """Write the weights to binary files."""
        if os.path.exists(folder):
            shutil.rmtree(folder)
            
        os.makedirs(folder, exist_ok=True)
        for layer, values in quantized_params.items():
            write_array_to_memh_file(values[0], f"{layer}_weights", folder)
            write_array_to_memh_file(values[1], f"{layer}_biases", folder)
    
    def write_quantized_weights(self, folder):
        quantized_params = self.__quantize_parameters()
        for layer_name, values in quantized_params.items():
            # set only as many parameter tensors as layer requires, skips bias in Dense layer that were trained without them
            layer = self.__model.get_layer(layer_name)
            layer.set_weights(values[:len(layer.get_weights())])
        self.__write_weights(quantized_params, folder)