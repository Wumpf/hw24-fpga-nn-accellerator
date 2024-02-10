from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rerun as rr
from fxpmath import Fxp
import tensorflow as tf

# List all available devices
devices = tf.config.list_physical_devices()
print(f"Devices: {devices}")

rr.init("train_image", spawn=True)

# Load an animated gif and convert it to a sequence of images
# image_path = 'face2_input.gif'

# Load the image
image_path = 'face_training.png'
image = Image.open(image_path)
image = image.resize((64, 64))
width, height = image.size

# Convert the image to RGB (if it's not already in RGB)
image = image.convert('RGB')

# width = 128
# height = 128
# image = np.zeros((width, height, 3), dtype=np.uint8)
# for y in range(width):
#     image[:, y, 0] = np.linspace(0, 255, width)
# for x in range(height):
#     image[x, :, 1] = np.linspace(0, 255, height)


rr.log("train_input", rr.Image(image))

# Prepare input data (X): Normalized pixel coordinates
x_coords = np.arange(width).reshape(-1, 1) / width  # Normalize x coordinates
y_coords = np.arange(height).reshape(-1, 1) / height  # Normalize y coordinates

# rnd = np.random.rand(128)
#X = np.array([np.array([[x,y] for i in range(128)]).flatten()
#               for y in y_coords for x in x_coords])


joel = 128
# the sin functions introduce artifacts at the bottom
X = np.array([np.array([[np.sin(x * i), np.sin(y * i)] for i in range(joel)]).flatten()
               for y in y_coords for x in x_coords])

rr.log("log", rr.TextLog(str(X.shape)))
#rr.log("input_vec", rr.Tensor(np.reshape(X, (256, 128, 128))))

# Prepare output data (Y): Normalized RGB values
pixels = np.array(image) / 255.0  # Normalize pixel values
Y = pixels.reshape(-1, 3)  # Reshape to a 2D array where each row is RGB

print(f"X.shape : {X.shape}")
print(f"Y.shape : {Y.shape}")

layer_size = 64
model = Sequential([
    Dense(layer_size, input_dim=256, activation='relu'),
    Dense(layer_size, activation='relu'),
    #Dense(layer_size, activation='relu'),
    Dense(3, activation='sigmoid')  # Use 'sigmoid' to output values between 0 and 1
])

model.compile(optimizer='nadam', loss='mean_squared_error')
history = model.fit(X, Y, epochs=100, batch_size=32, verbose=True, validation_split=.1)  # Adjust epochs and batch size as needed

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

predicted_rgb = model.predict(X) * 255  # Rescale the output
predicted_rgb = predicted_rgb.reshape((height, width, 3)).astype(np.uint8)

# Create and save the generated image
generated_image = Image.fromarray(predicted_rgb)
# generated_image.save('training_output.png')
# generated_image.show()

# Plot the training and validation loss
# plt.figure(figsize=(16, 8))
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='validation loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()


rr.log("train_output", rr.Image(generated_image))