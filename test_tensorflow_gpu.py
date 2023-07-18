# %%
import tensorflow as tf
from tensorflow.python.client import device_lib
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.is_built_with_cuda())
    print(device_lib.list_local_devices())

# [name: "/device:CPU:0"
#  device_type: "CPU"
#  ...
#  name: "/device:GPU:0"
#  device_type: "GPU"
# ...]