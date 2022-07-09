from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model

import numpy as np
import matplotlib.pyplot as plt
import time


start = time.time()

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# from sparse label to categorical
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# use functional API to build cnn layers
inputs = Input(shape=input_shape)

y = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
# image to vector before connecting to dense layer
y = Flatten()(y)
# dropout regularization
y = Dropout(dropout)(y)

outputs = Dense(num_labels, activation='softmax')(y)

# build the model by supplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)

# network model in text
model.summary()
plot_model(model, to_file='./code/chap02/cnn_functional.png', show_shapes=True)

# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# train the model with input images and labels
history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test),
                    epochs=20,
                    batch_size=batch_size)

# model accuracy on test dataset
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size,
                       verbose=0)

print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))

print(f"elapsed time: {time.time() - start}")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
