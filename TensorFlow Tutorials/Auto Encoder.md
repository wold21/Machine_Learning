# Auto Encoder

입력을 출력에 복사하도록 훈련 된 특수한 유형의 신경망이다.

예를 들면 손으로 쓴 숫자의 이미지가 주어지면 오토 인코더는 먼저 이미지를 더 낮은 차원의 잠재 표현으로 인코딩 한 다음(pooling) 잠재표현을 다시 디코딩 한다. 오토 인코더는 재구성 오류(pooling된 데이터를 다시 되돌릴 때)를 최소화하면서 데이터를 압축하는 방법을 학습하는 신경망이다.

## 텐서플로우 및 기타 라이브러리 가져오기

~~~python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
~~~

- 데이터 세트 로드

~~~python
(X_train, _), (X_test, _) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

print(X_train.shape)
print(X_test.shape)
~~~

~~~python
(60000, 28, 28)
(10000, 28, 28)
~~~

### 첫번째 예시 : 기본 오토 인코더

이미지를 64차원 잠재 벡터로 압축하는 인코더 및 잠재 공간에서 이미지를 재구성하는 디코더라는 두개의 Dense 레이어로 오토 인코더를 정의한다.

~~~python
latent_dim = 64
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
                                            layers.Flatten(),
                                            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
                                            layers.Dense(784, activation='sigmoid'),
                                            layers.Reshape((28, 28))
        ])


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(latent_dim)
~~~

~~~python
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
~~~

X_train을 입력과 목표 모두로 사용하여 모델을 훈련시킵니다. 인코더는 데이터 세트를 784차원에서 잠재 공간으로 압축하는 방법을 배우고 디코더는 원본이미지를 재구성하는 방법을 배웁니다.

~~~python
autoencoder.fit(X_train, X_train,
                epochs=10,
                shuffle=True,
                validation_data=(X_test, X_test))

1875/1875 [==============================] - 4s 2ms/step - loss: 0.0088 - val_loss: 0.0089
~~~

모델 학습 완료 테스트 데이터로 테스트 해보기

~~~python
encoded_imgs = autoencoder.encoder(X_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
~~~

~~~python
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstruction")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
~~~

![2](C:\Users\USER\Desktop\Workspace\doc\2.PNG)

### 두번째 예시 :  노이즈 제거

오토인코더의 장점으로 노이즈 제거를 훈련할 수 있다.

~~~python
# 위의 결과를 덮을 데이터 세트 다시 가져오기
(X_train, _), (X_test, _) = fashion_mnist.load_data()
X_train.shape, X_test.shape

((60000, 28, 28), (10000, 28, 28))
~~~

~~~python
# 데이터를 실수형태로 만들어주는 이유는 범위를 0~1사이로 만들기 위해 
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# 채널 정보 추가하기
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

X_train.shape, X_test.shape

((60000, 28, 28, 1), (10000, 28, 28, 1))
~~~

- 이미지에 임의의 노이즈 추가

~~~python
# 노이즈 값
noise_factor = 0.2

# X_train, X_test에 노이즈를 표준분포화 시켜 더해준다. 
X_train_noisy = X_train + noise_factor * tf.random.normal(shape=X_train.shape)
X_test_noisy = X_test + noise_factor * tf.random.normal(shape=X_test.shape)

# 잘모르겠지만 제일 큰 값은 노이즈 걸린 데이터를 0.과 1.로만 구분지어 만든다.
# 텐서를 지정된 값으로 자른다.
# https://www.tensorflow.org/api_docs/python/tf/clip_by_value
X_train_noisy = tf.clip_by_value(X_train_noisy, clip_value_min=0., clip_value_max=1.)
X_test_noisy = tf.clip_by_value(X_test_noisy, clip_value_min=0., clip_value_max=1.)
~~~

- 잡음이 있는 이미지를 플로팅한다.

~~~python
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1) # 행의수, 열의수, 인덱스 -> 각 정수는 10보다 작아야한다.
    plt.title("original+ noise")
    plt.imshow(tf.squeeze(X_test_noisy[i]))  # 차원 축소를 하는 이유? 모든 데이터가 아니기 때문? (10000, 28, 28, 1) -> (28, 28, 1)
    plt.gray()
plt.show()
~~~

![3](C:\Users\USER\Desktop\Workspace\doc\capture\3.PNG)

### CAE 정의

이 예제에서는 train convolution auto encoder를 훈련한다. conv2d 레이어는 인코더 conv2dtranspose레이어는 디코더

~~~python
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)), 
      layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)])
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
autoencoder = Denoise()
~~~

- 잘 학습된 인코더 모델은, 데이터의 특징 추출기로 사용 가능.

~~~python
autoencoder.compile(optimizer='adam', loss=losses.MeanAbsoluteError())
~~~

~~~python
autoencoder.fit(X_train_noisy, X_train,
                epochs=10,
                shuffle=True,
                validation_data=(X_test_noisy, X_test))

Epoch 10/10
1875/1875 [==============================] - 68s 36ms/step - loss: 0.0460 - val_loss: 0.0460
~~~

- 인코더 요약을 살펴보자 이미지가 7,7로 다운샘플링 되었다.

~~~python
autoencoder.encoder.summary()

Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 14, 14, 16)        160       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 8)           1160      
=================================================================
Total params: 1,320
Trainable params: 1,320
Non-trainable params: 0
_________________________________________________________________
~~~

- 디코더는 이미지를 다시 7, 7에서 28, 28로 업샘플링함.

~~~python
autoencoder.decoder.summary()

Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_transpose (Conv2DTran (None, 14, 14, 8)         584       
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 16)        1168      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 1)         145       
=================================================================
Total params: 1,897
Trainable params: 1,897
Non-trainable params: 0
_________________________________________________________________
~~~

~~~python
encoded_imgs = autoencoder.encoder(X_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
~~~

~~~python
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origin + noise
    ax = plt.subplot(2, n, i +1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(X_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.title("reconstruction")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display original
    ax = plt.subplot(2, n, i + n + 2)
    plt.title("original")
    plt.imshow(tf.squeeze(X_test[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
~~~

![4](C:\Users\USER\Desktop\Workspace\doc\capture\4.PNG)

![5](C:\Users\USER\Desktop\Workspace\doc\capture\5.PNG)

- 노이즈는 아주 잘 제거가 되었지만 원본에 비하면 손상이 있다 하지만 특성은 나름 잘 잡아낸거 같다.