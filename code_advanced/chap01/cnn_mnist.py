from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import time


start = time.time()

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))
print(f"num_labels   : {num_labels}")
print(f"type(y_train): {type(y_train)}")
print(f"y_train.shape: {y_train.shape} \n")
print(f"type(y_test) : {type(y_test)}")
print(f"y_test.shape : {y_test.shape} \n")

# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f"type(y_train): {type(y_train)}")
print(f"y_train.shape: {y_train.shape} \n")
print(f"type(y_test) : {type(y_test)}")
print(f"y_test.shape : {y_test.shape} \n")

# image dimensions (assumed square)
image_size = x_train.shape[1]
input_size = image_size * image_size
print(f"image_size: {image_size}")
print(f"input_size: {input_size}")

# resize and normalize
image_size = x_train.shape[1]
# resize and normalize
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
# image is processed as is (square grayscale)
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2

# model is a stack of CNN-ReLU-MaxPooling
model = Sequential()
model.add(Conv2D(filters=filters, 
                 kernel_size=kernel_size, 
                 activation='relu', 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(Flatten())

# dropout added as regularizer
model.add(Dropout(dropout))

# output layer is 10-dim one-hot vector
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()
plot_model(model, to_file='./code/chap01/cnn_mnist.png', show_shapes=True)

# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# train the network
history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

# validate the model on test dataset to determine generalization
_, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

print("\nTest accuracy: %.1f%%" % (100.0 * acc))

print(f"elapsed time: {time.time() - start}")

acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()

plt.show()
