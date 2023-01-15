# %% class Vectorizer
import string

class Vectorizer:
    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def tokenize(self, text):
        return text.split()

    def make_vocabulary(self, dataset):
        self.vocabulary = {"": 0, "[UNK]": 1}
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict((v, k) for k, v in self.vocabulary.items())

    # TODO self.vocabulary.get 
    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return " ".join(self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)

# %% main
vectorizer = Vectorizer()
dataset = ["I write, erase, rewrite",
           "Erase again, and then",
           "A poppy blooms.",]
vectorizer.make_vocabulary(dataset)

# %%
print("vocabulary")
print(vectorizer.vocabulary)
print()

print("inverse_vocabulary")
print(vectorizer.inverse_vocabulary)

# %%
test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = vectorizer.encode(test_sentence)
print(encoded_sentence)

# %%
decoded_sentence = vectorizer.decode(encoded_sentence)
print(decoded_sentence)

# %%
from tensorflow.keras.layers import TextVectorization

text_vectorization = TextVectorization(output_mode="int",)

import re
import string
import tensorflow as tf

def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(lowercase_string, 
                                    f"[{re.escape(string.punctuation)}]", "")

def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)

text_vectorization = TextVectorization(output_mode="int",
                                       standardize=custom_standardization_fn,
                                       split=custom_split_fn,)

# %%
dataset = ["I write, erase, rewrite",
           "Erase again, and then",
           "A poppy blooms.",]

text_vectorization.adapt(dataset)

# %%
text_vectorization.get_vocabulary()

# %%
vocabulary = text_vectorization.get_vocabulary()
test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorization(test_sentence)
print(encoded_sentence)

# %%
inverse_vocab = dict(enumerate(vocabulary))
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
print(decoded_sentence)