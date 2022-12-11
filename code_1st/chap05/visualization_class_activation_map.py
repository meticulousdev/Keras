# %%
# from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os

import cv2


tf.compat.v1.disable_eager_execution()

# %%
model = VGG16(weights='imagenet')

# %%
folder_dir = os.getcwd()
# folder_dir = './code/chap05'
img_path = folder_dir + '/datasets/creative_commons_elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print(f"x.shape (img_to_array)    : {x.shape}")
print()

x = np.expand_dims(x, axis=0)
print(f"x.shape (np.expand_dims)  : {x.shape}")
print(f"np.max(x) : {np.max(x)}")
print(f"np.min(x) : {np.min(x)}")
print()

x = preprocess_input(x)
print(f"x.shape (preprocess_input): {x.shape}")
print(f"np.max(x) : {np.max(x)}")
print(f"np.min(x) : {np.min(x)}")
print()

# %%
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
print(f"np.argmax(preds[0]): {np.argmax(preds[0])}")
# np.argmax(preds[0]): 386

# %%
# 예측 벡터의 '아프리카 코끼리' 항목
african_elephant_output = model.output[:, 386]

# VGG16의 마지막 합성곱 층인 block5_conv3 층의 특성 맵
last_conv_layer = model.get_layer('block5_conv3')

# block5_conv3의 특성 맵 출력에 대한 '아프리카 코끼리' 클래스의 그래디언트
grads = backend.gradients(african_elephant_output, last_conv_layer.output)[0]

# 특성 맵 채널별 그래디언트 평균 값이 담긴 (512,) 크기의 벡터
pooled_grads = backend.mean(grads, axis=(0, 1, 2))

# 샘플 이미지가 주어졌을 때 방금 전 정의한 pooled_grads와 block5_conv3의 특성 맵 출력을 구합니다
iterate = backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# 두 마리 코끼리가 있는 샘플 이미지를 주입하고 두 개의 넘파이 배열을 얻습니다
pooled_grads_value, conv_layer_output_value = iterate([x])

# "아프리카 코끼리" 클래스에 대한 "채널의 중요도"를 특성 맵 배열의 채널에 곱합니다
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# 만들어진 특성 맵에서 채널 축을 따라 평균한 값이 클래스 활성화의 히트맵입니다
heatmap = np.mean(conv_layer_output_value, axis=-1)

# %%
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

# %%
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

cv2.imwrite('./datasets/elephant_cam.jpg', superimposed_img)