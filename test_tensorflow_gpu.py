# %%
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# [name: "/device:CPU:0"
#  device_type: "CPU"
#  ...
#  name: "/device:GPU:0"
#  device_type: "GPU"
# ...]