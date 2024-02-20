import rerun as rr

from PIL import Image

from common import NeuralNetwork

rr.init("train_image", spawn=True)

if True:
    mode = 'RGBA'
    channels = 4
else:
    mode = 'RGB'
    channels = 3

# Load the image
image_path = 'pretty_256.png'
#image_path = 'face_input.png'
#image_path = 'training_input.png'
image = Image.open(image_path)
image = image.resize((64, 64))
image = image.convert(mode)
width, height = image.size

rr.log("train_input", rr.Image(image))

embedding_size = 32
layer_size = 64

# nn = NeuralNetwork(width, height, layer_size, embedding_size, channels, mixed_precision="mixed_float16")
nn = NeuralNetwork(width, height, layer_size, embedding_size, channels)

history = nn.train(image, 200, 32)

predicted_rgba = nn.predict()
generated_image = Image.fromarray(predicted_rgba, mode)

rr.log("train_output", rr.Image(generated_image))

nn.write_quantized_weights("output")

q_predicted_rgba = nn.predict()
q_generated_image = Image.fromarray(q_predicted_rgba, mode)

rr.log("quantized_output", rr.Image(q_generated_image))
