import rerun as rr

from PIL import Image

from common import NeuralNetwork

rr.init("train_image", spawn=True)

# Load the image
image_path = 'pretty_256.png'
image = Image.open(image_path)
image = image.resize((64, 64))
image = image.convert('RGB')
width, height = image.size

rr.log("train_input", rr.Image(image))

embedding_size = 32
layer_size = 64
nn = NeuralNetwork(width, height, layer_size, embedding_size)

history = nn.train(image, 100, 32)

nn.write_quantized_weights("output")

predicted_rgb = nn.predict()

generated_image = Image.fromarray(predicted_rgb)

rr.log("train_output", rr.Image(generated_image))