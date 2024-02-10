import tensorflow as tf
print(tf.__version__)
import keras as K
from keras import layers
print(K.__version__)
import matplotlib.pyplot as plt
import numpy as np

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
print("Training data shape: ", x_train.shape) 
print("Test data shape", x_test.shape) 

# Flatten the images
image_size = 784 # 28*28
x_train = x_train.reshape(x_train.shape[0], image_size)
x_test = x_test.reshape(x_test.shape[0], image_size)

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = K.utils.to_categorical(y_train, num_classes)
y_test = K.utils.to_categorical(y_test, num_classes)

# build a classical Neural Network using two dense layers of 16 units per layer, 
# sigmoid activation for hidden-layers and softmax for output layer.
model = K.Sequential()
model.add(K.layers.Dense(units=32, activation='sigmoid', input_shape=(image_size,), name='layer_0'))
model.add(K.layers.Dense(units=32, activation='sigmoid', name='layer_1'))
model.add(K.layers.Dense(units=num_classes, activation='softmax', name='output'))
model.summary()

#
# Model compiling and training
#

# Evaluate performance on test partition of the dataset:
N_epochs = 50
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=N_epochs, verbose=True, validation_split=.1)

# Plot the training evolution vs epochs
plt.figure(figsize=(16,8))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('model accuracy') 
plt.ylabel('accuracy')
plt.xlabel('epoch') 
plt.legend()
plt.ylim([0.4, 1])
plt.show()