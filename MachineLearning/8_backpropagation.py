# -*- coding: utf-8 -*-
"""8_Backpropagation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OAbXPBSs5u-jryT6XJB3TTJt1E4moy8X
"""

import numpy as np

# 곱셈 계층 구현
class MulLayer:

    # 초기화
    # 곱셈 계층으로 들어오고 있는 값을 저장할 변수 만들기
    # -> x 방향으로 갈 때는 dout * y
    # -> y 방향으로 갈 때는 dout * x
    def __init__(self):
        self.x = None
        self.y = None


    # 곱셈 레이어
    # 노드로 들어온 값에 대한 곱을 수행한다.
    def forward(self, x, y):
        self.x = x
        self.y = y 
        out = x * y

        return out


    # dout 미분 다음 노드에서 넘어온 노드값.
    # 상류에서 역전파된 미분값(dout)에 흘러 들어온 x, y를 각각 바꿔서 곱한 값을 리턴
    def backward(self, dout):
        # 무조건 미분값이 리턴된다.
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy

# 순전파를 이용해 사과 가격 확인하기
apple_price = 100
apple_cnt = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 먼저 사과 가격 곱하기 사과 개수의 결과를 구해야한다.
total_apple_price = mul_apple_layer.forward(apple_price, apple_cnt)
final_apple_price = mul_tax_layer.forward(total_apple_price, tax)

final_apple_price

# 역전파 구현하기
# 마지막 출력층에 대한 미분값 준비 ( 1 )
dprice = 1 # 출력층의 결과물에 대한 미분값

# tax_layer 부터 시작 -> apple_layer
dtotal_apple_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dapple_cnt = mul_apple_layer.backward(dtotal_apple_price)

# 개별 사과의 미분값, 사과 개수의 미분값, 소비세의 미분값
dapple_price, dapple_cnt, dtax

# 덧셈 계층 구현
class AddLayer:
    # 덧셈 계층의 특징
    # 역전파 때 미분값이 들어오면 그대로 입력 신호에 흘려보내준다.
    # 그래서 초기화 과정에 변수로 입력된 값에 대한 값을 저장할 필요는 없다.
    def __init__(self):
        pass

    
    def forward(self, x, y):
        out = x + y
        
        return out


    def backward(self, dout):
        # 곱셈 계층의 코드 모양과 비슷하게 해주기 위해서 1을 곱했음.
        dx = dout * 1
        dy = dout * 1

        return dx, dy

# 순전파 역전파 구현

# 노드애 들어갈 입력신호
apple_price = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층 만들기 각 과일의 개수 * 가격
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()

# 각 과일의 가격들을 더해주는 레이어
add_apple_orange_layer = AddLayer()

# 소비세를 적용시키는 레이어
mul_tax_layer = MulLayer()

# 순전파 수행 
apple_price = mul_apple_layer.forward(apple_price, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
total_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(total_price, tax)

# 역전파 수행하기
dprice = 1
dtotal_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dtotal_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num   = mul_apple_layer.backward(dapple_price)

print("사과 2개 오렌지 3개의 가격 : {}".format(price))
print("사과 개수 미분 : {}".format(dapple_num))
print("사과 가격 미분 : {}".format(dapple))
print("오렌지 개수 미분 : {}".format(dorange_num))
print("오렌지 가격 미분 : {}".format(dorange))
print("소비세 미분 : {}".format(dtax))

"""## 신경망을 위한 각종 계층 구현하기
1. 활성화 함수 계층 구현하기
    - ReLU
    - Sigmoid

\# 02.역전파 구현해보기.html

## ReLU
"""

# ReLU
# 음수는 0으로 통일, 양수는 양수 그대로 사용
x = np.arange(-5, 6)

# 마스킹 과정
# 배열이 True False
mask = (x <= 0)
x_relu = x.copy()
x_relu[mask] = 0
x_relu

class ReLU:

    def __init__(self):
        self.mask = None


    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy() # 원본 배열 복사
        out[self.mask] = 0

        return out

    # backward로 들어오는 미분값이 들어있는 배열은
    # forward때와는 다르게 원래 0이었던 곳에 0이 아닌 값이 들어갈 수도 있다.
    # 따라서 원래 0이었던 곳( mask )은 전부 0으로 다시 바꿔 준다.
    def backward(self, dout):

        # 미분된 배열에다가도 마스킹 처리를 해준다.
        dout[self.mask] = 0
        dx = dout
        
        return dx

"""## Sigmoid"""

# 시그모이드 계층 만들기
# 시그모이드 함수가 y라면, 시그모이드 함수를 미분했을 때는 y(1-y)이다.
# 즉, 순전파 진행 시에 y의 결과값을 저장하고 있으면, 이 값을 그대로 역전파 때 활용할 수 있다는 이야기

class Sigmoid:

    # 순전파 할 때 계산되었던 값을 가지고 있는다.
    def __init__(self):
        self.out = None


    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out


    def backward(self, dout):
        dx = dout(1.0 - self.out) * self.out
        
        return dx

"""## Affine / Softmax 계층 구현하기"""

# 행렬곱의 역전파는 상대 계수의 전치 행렬을 하여 흘러들어온 값을 곱한다.
# Affine 행렬의 내적

# 전치 행렬
X = np.arange(6).reshape(2, 3)
X

X.T

class Affine:


    # 초기 가중치, 초기 편향 가지고 있기
    def __init__(self, W, B):
        self.W = W
        self.B = B

        self.X = None
        self.original_X_shape = None # 원본 데이터의 모양을 유지하기 위한 변수

        # 경사하강법을 통해 가중치를 업데이트 해야해서, 기존 미분값을 알고 있어야한다.
        # 히스토리 남기기
        self.dW = None
        self.db = None
    
    
    def forward(self, x):
        # 데이터의 원본 형상 저장.
        # 3차원이상의 텐서에 대응하기 위해 1차원으로 만들어 줌.
        original_X_shape = X.shape
        X = X.reshape(x.shape[0], -1) # 데이터를 1차원으로 만들어준다.

        self.X = X

        out = np.dot(self.X, self.W) + self.B

        return out
    
    
    def backward(self, dout):
        # X의 입력신호 방향으로 흘러나가는 값은 미분값 * W.T
        dx = np.dot(dout, self.W.T)

        # W의 입력 신호 방향으로 흘러나가는 값은 입력값의 전치행렬 * 미분값
        self.dW = np.dot(self.X.T, dout)

        # 유닛마다 편향이 다르기 때문에 dout과 편향값을 그냥 더하면 된다.
        # 차원 맞춰주려고 더했다.
        self.db = np.sum(dout, axis=0) 

        # 튜플 안의 값이 들어가야하기때문에 *를 붙혀 언팩킹을 했다.
        dx = dx.reshpe(*self.original_X_shape)

        return dx

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size # 데이터 1개당 오차를 앞 계층으로 전파함
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1 # 원-핫 인코딩이 안되어있을 때
            dx = dx / batch_size
        
        return dx

# %cd /content/common
# ! unzip common.zip

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1 # 마지막 계층의 미분값 설정
        dout = self.lastLayer.backward(dout) # 마지막 계층에서의 미분값 전달 받기 (SoftMaxWithLoss에서 받음)
        
        layers = list(self.layers.values()) # 저장된 레이어를 불러와서 ( 여기서는 순차적인 레이어가 저장 되어 있음 )
        layers.reverse() # 뒤집음(뒤에서 부터 전달해야 하기 때문에 )
        
        # 뒤에서 부터 역전파
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

import sys, os
sys.path.append(os.pardir)
import numpy as np
from tensorflow.keras import datasets
mnist = datasets.mnist

# 데이터 읽어오기
(X_train, t_train), (X_test, t_test) = mnist.load_data()

network = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)

X_batch = X_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(X_batch, t_batch)
grad_backpop   = network.gradient(X_batch, t_batch)

# 각 가중치의 차이의 절댓값을 구하고 그 절댓값들의 평균을 낸다.
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backpop[key] - grad_numerical[key]))
    print(key + " : {}".format(diff))

