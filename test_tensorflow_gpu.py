# %%
import tensorflow as tf
from tensorflow.python.client import device_lib


if __name__ == "__main__":
    print(tf.__version__)
    print(tf.test.is_built_with_cuda())
    print(device_lib.list_local_devices())

# [name: "/device:CPU:0"
#  device_type: "CPU"
#  ...
#  name: "/device:GPU:0"
#  device_type: "GPU"
# ...]