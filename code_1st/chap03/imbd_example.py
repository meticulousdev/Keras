# %%
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import losses, metrics
import numpy as np
import matplotlib.pyplot as plt


# %%
# ?? It has already been preprocessed: 
# the reviews (sequences of words) have been turned into sequences of integers, 
# where each integer stands for a specific word in a dictionary.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# %%
print("encoded review")
print(f"type(train_data) : {type(train_data)}")
print(f"train_data.shape : {train_data.shape}")
print(f"train_labels[0]  : {train_labels[0]}")
print(f"train_data[0]    : {train_data[0]}")
print()

print("decoded review")
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join([reverse_word_index.get(i - 3, "\n") for i in train_data[0]])
print(f"{decoded_review}")
print()

# %% one-hot encoding
# ?? Can repetition of the same word be considered?
# ?? dimension=10000
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# %%
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(f"type(x_train)    : {type(x_train)}")
print(f"x_train.shape    : {x_train.shape}")

# %%
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# %%
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# %%
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# %%
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# %%
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# %%
history_dict = history.history
print(f"history dict: {history_dict.keys()}")

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# %%
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()

# %%
results = model.evaluate(x_test, y_test)
print(f"test loss, test accuracy : {results}")

# %%
history = model.fit(x_train,
                    y_train,
                    epochs=4,
                    batch_size=512)

# %%
results = model.evaluate(x_test, y_test)
print(f"test loss, test accuracy : {results}")
