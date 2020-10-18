# -*- coding: utf-8 -*-
"""10_Tensorflow를 이용한 시퀀셜 모델링.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_R5ePKEYRtHiOczKbza_Jja6jGNNMAM5
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

"""# TF-1"""

import numpy as np
x = np.arange(-1.0, 5.0)
y = np.arange(5.0, 11.0)

x

y

# 시퀀셜 모델
model = Sequential([
                    Dense(1),            
])

# 모델 생성
model.compile(optimizer='sgd', loss='mse')
model.fit(x, y, epochs=1200)

model.predict([10.0])

weight=model.get_weights()

weight[0], weight[1]

"""# TF-2 Fashion MNIST"""

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

X_train.shape

plt.imshow(X_train[3000], 'gray')
plt.show()

# 리스케일링
X_train, X_test = X_train / 255., X_test / 255.

"""## DNN으로 이미지 문제 풀기"""

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
                    Flatten(input_shape=(28, 28)),
                    Dense(128, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

"""## ModelCheckPoint 만들기
- 1에폭당 훈련을 하게 되면 loss, acc등이 나온다.
- 1에폭 훈련시에 제일 좋았던 가중치와 편향 등을 저장할 수 있게 해준다.
"""

# 체크 포인트 파일 이름 만들기
checkpoint_path = 'my_checkpoint.ckpt'
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True, 
    save_best_only=True,
    monitor='val_loss', # 검증 세트에 대한 정확도만 살핀다.
    verbose=1 # 로깅 설정
)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test), # 원래는 검증데이터로 해야하는데 여기서는 그냥 테스트 데이터로 함.
          epochs=10,
          callbacks=[checkpoint])
model.load_weights(checkpoint_path) # 최고의 효과를 내었던 가중치를 불러오기

"""# 테스트 세트로 evaluate 할 것!
근데 우리는 테스트 세트를 검증 세트로 써버림. 

실제로는 이 단계에 테스트 세트를 사용해야한다.

# TF-3 중에 하나 CAT VS DOG
"""

import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

dataset_name = 'cats_vs_dogs'

train_dataset, info = tfds.load(name=dataset_name, split='train[:80%]', with_info=True)
val_dataset, info = tfds.load(name=dataset_name, split='train[-20%:]', with_info=True)

"""이미지 리스케일링 및 전처리"""

def preprocess(features):
    img, lbl = tf.cast(features['image'], tf.float32) / 255., features['label']

    # 이미지 사이즈 조절 tf.image 기능을 사용한다.
    # 데이터 셋의 size의 값은 쌤이 알아낸 적정값임.
    img = tf.image.resize(img, size=(224, 224))

    return img, lbl

"""### 데이터 전처리"""

batch_size = 32

# map : 자료구조에서 데이터를 하나씩 꺼내서 특정 함수를 적용시킨다.
train_data = train_dataset.map(preprocess).batch(batch_size) # 전처리 후 배치로 묶기
val_data = val_dataset.map(preprocess).batch(batch_size) # 전처리 후 배치로 묶기

total_image_size = info.splits['train'].num_examples
steps_per_epoch = int(total_image_size * 0.8) // batch_size + 1
val_steps = int(total_image_size * 0.2) // batch_size + 1

model = Sequential([
                    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
                    MaxPooling2D(2, 2),
                    Conv2D(32, (3, 3), activation='relu'),
                    MaxPooling2D(2, 2),
                    Conv2D(16, (3, 3), activation='relu'),
                    MaxPooling2D(2, 2),
                    Flatten(),
                    Dropout(0.3),
                    Dense(512, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(2, activation='softmax')
])

"""RMSProp사용하기
- 학습률을 학습 도중에 바꿀 수 있다.
"""

# 현재 학습률에 대해 val_loss가 epoch를 3번 도는데 나아지지 않으면 학습률을 조정한다. (RMSProp과 같이 사용됨)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    min_lr = 0.00001
)
optimizer = tf.keras.optimizers.RMSprop(0.001)

"""## 모델 컴파일"""

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

"""## 모델 훈련"""

checkpoint_path = "cat_dog_checkpoint.ckpt"
checkpoint = ModelCheckpoint(
    checkpoint_path,
    save_best_only = True,
    save_weights_only = True,
    monitor = 'val_loss',
    verbose = 1
)

model.fit(
    train_data,
    validation_data=(val_data),
    batch_size=32,
    epochs=40, # train에서 
    callbacks=[checkpoint, reduce_lr]
)
model.load_weights(checkpoint_path)

