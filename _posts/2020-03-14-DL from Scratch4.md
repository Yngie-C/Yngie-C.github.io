---
layout: post
title: 4. 오차역전파법
category: DL from Scratch
tag: Deep-Learning
---



## 1) 계산 그래프

- 계산 그래프 : 계산 과정을 노드(Node)와 에지(Edge)를 사용하여 표현한다. 노드는 두 신호의 합을 더하는 덧셈 노드와 두 신호를 곱하는 곱셈 노드로 이루어져 있다.



- 계산 그래프의 계산 방식

1. 계산 그래프를 구성한다.
2. 그래프에서 계산을 왼쪽에서 오른쪽으로 진행한다. 이 과정을 **순전파(Forward Propagation)** 라고 한다.



- **역전파(Back Propagation)** : 계산 그래프에서 순전파의 반대 방향(오른쪽에서 왼쪽)으로 진행하는 계산이다. 역전파는 국소적 미분값을 전달한다. 아래는 계산 그래프에서 순전파(초록색)와 역전파(빨간색)가 진행되어 가는 과정을 그림으로 나타낸 것이다. 순전파는 $(x, y, z)$ 에서 각각 $-2, 5, -4$ 의 신호를 받아 덧셈과 곱셈 노드를 통해 $-12$ 라는 최종 값을 산출해내게 된다. 역전파는 $1$ 로 시작하여 덧셈 노드가 각 신호에게 원래 값을 그대로 전달하며, 곱셈 노드는 각 신호에게 맞은편 신호에서 받은 만큼을 돌려준다.  

<p align="center"><img src="https://miro.medium.com/max/936/1*6oUtr7ENFHOK7J4lXJtw1g.png" alt="back prop" style="zoom:50%;" /></p>



## 2) 연쇄 법칙과 역전파

- **연쇄법칙** : 위에서 살펴본 역전파에서는 국소적인 미분을 오른쪽에서 왼쪽으로 전달한다. 이렇게 할 수 있는 이유는 연쇄법칙(Chain rule)을 따른 것이다. 연쇄법칙은 "합성 함수의 미분을 합성 함수를 구성하는 각 함수 미분의 곱으로 나타낼 수 있다"이며 아래의 예를 통해 수식으로 살펴볼 수 있다. 

$$
z = t^2,\quad t = x + y \text{ 는 } z = (x + y)^2 \text{ 이므로} \\
\text{연쇄법칙을 이용하여 x에 대한 z의 미분 값을 구하면,} \\
\frac{\partial{z}}{\partial{x}} = \frac{\partial{z}}{\partial{t}} \frac{\partial{t}}{\partial{x}} \text{ 와 같이 나타낼 수 있다.}
$$



- 연쇄 법칙과 계산 그래프 : 아래는 이전에 계산 그래프에서의 노드를 $f(x)$ 로 함수화하여 나타낸 것이다. 순전파에서 어떤 노드가 $f(x,y) = z$ 일 때, 역전파는 연쇄법칙을 이용해 오른쪽과 같이 나타낼 수 있다. 

![chain rule](https://camo.githubusercontent.com/7881f7e56a9c507f43066b499e583aac3f423b18/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a71314d374c47694454697277552d344c634671375f512e706e67)



<br/>

## 3) 단순한 계층 구현하기

순전파와 역전파의 과정을 파이썬 코드로 구현해보자. 아래는 곱셈 노드와 덧셈 노드 중 곱셈 노드를 파이썬 코드로 구현한 것이다.

- 곱셈 계층

```python
class MulLayer:
    def __init__(self):		# x와 y를 초기화하기
        self.x = None
        self.y = None

    def forward(self, x, y):	# 순전파에서 두 신호의 곱을 구현
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y		# x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy
```

- 덧셈 계층 : 아래는 덧셈 노드를 파이썬 코드로 구현한 것이다.

```python
class AddLayer:
    def __init__(self):		# 초기화가 필요 없다.
        pass

    def forward(self, x, y):	# 입력 받은 두 신호의 합을 구현
        out = x + y

        return out

    def backward(self, dout):	# 역전파의 덧셈 노드에서는 두 신호를 그대로 나누어준다.
        dx = dout * 1
        dy = dout * 1

        return dx, dy
```



<br/>

## 4) 활성화 함수 계층 구현하기

- ReLU 계층 구현하기 : 활성화 함수에서 사용되는 ReLU의 수식은 다음과 같다. ReLU 수식 아래에 있는 것은 x에 대한 y의 미분을 식으로 나타낸 것이다.

$$
y = \begin{cases} x \quad (x > 0) \\ 0 \quad (x \leq 0) \end{cases} \\
\frac{\partial{y}}{\partial{x}} = \begin{cases} 1 \quad (x > 0) \\ 0 \quad (x \leq 0) \end{cases}
$$

위 수식을 파이썬 코드로 구현하면 다음과 같다.

```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        # mask 는 x > 0 일 때는 True, x <= 0 일 때는 False인 Bool값을 나타낸다. 
        out = x.copy
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```



- Sigmoid 계층 구현하기 : 활성화 함수에서 사용되는 시그모이드 함수의 수식은 다음과 같고, 그 아래의 수식은 시그모이드 함수를  x에 대해  y를 미분한 것이다.

$$
y = \frac{1}{1 + \exp{(-x)}} \\
\frac{\partial{y}}{\partial{x}} = y(1 - y)
$$

 위 수식을 파이썬 코드로 구현하면 다음과 같다.

```python
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out		#(1-y)y 형태를 구현한다.

        return dx
```



<br/>

## 5) Affine/Softmax 계층 구현하기

- Affine 계층 : 신경망의 순전파 때 수행하는 행렬의 곱을 기하학에서는 어파인 변환(Affine Transformation)이라고 한다. 그래서 이 책에서는 이러한 과정의 층을 Affine 계층이라고 명명한다. Affine 계층에서 일어나는 순전파와 역전파를 수식으로 나타내면 다음과 같다.

$$
\text{순전파 :} \quad
\mathbf{Y} = \mathbf{W} \cdot \mathbf{X} + \mathbf{B} \\
\text{역전파 :} \quad \frac{\partial{L}}{\partial{\mathbf{X}}} = \frac{\partial{L}}{\partial{\mathbf{Y}}} \cdot \mathbf{W}^T \qquad \frac{\partial{L}}{\partial{\mathbf{W}}} = \mathbf{X}^T \cdot \frac{\partial{L}}{\partial{\mathbf{Y}}}
$$



- 배치용 Affine 계층 : 순전파 과정을 하나의 데이터로만 수행하는 것이 아니라, N개의 데이터로 하는 과정이다. $\mathbf{X}$ 가 1차원 텐서(벡터)가 아닌 2차원 텐서 $(N, n)$ 으로 주어진다. (배치용) Affine 계층을 구현하는 파이썬 코드는 아래와 같다.

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
```



- Softmax-with-Loss 계층 : Softmax 층 내에 손실 함수인 크로스엔트로피 함수를 추가하여 Softmax-with-Loss  함수를 구현해본다.
  - Softmax 계층 : 입력 신호인 $(a_1, a_2, a_3, ... , a_n)$ 을 정규화하여 예측값 $(y_1, y_2, y_3, ... , y_n)$ 을 출력한다. 여기서 $\sum^n_{I=1} y_i = 1$ 이다.
  - Cross Entropy Error 계층 : Softmax 함수의 출력값인 $(y_1, y_2, y_3, ... , y_n)$ 과 실제 레이블 $(t_1, t_2, t_3, ... , t_n)$ 을 입력 데이터로 받아 이 두 텐서로부터 손실 $L$ 을 출력하게 된다.
  - Softmax-with-Loss 계층의 역전파 : 이 계층에서의 역전파 결과는 $(y_1- t_1, y_2 - t_2, y_3 - t_3, ... , y_n - t_n)$ 이다. 이 계층에서의 순전파와 역전파를 파이썬 코드로 나타내면 아래와 같이 구현할 수 있다.

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
```



<br/>



## 6) 오차역전파법 구현하기

- 오차역전파법을 적용한 신경망 구현하기 : 2층으로 이루어진 신경망을 `TwoLayerNet` 클래스로 구현.

```python
# coding: utf-8
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
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
```



- 오차역전파법으로 구한 기울기 검증하기 : 수치 미분은 구현하기 쉽지만 느리다. 반면 해석적으로 수식을 풀어내는 오차역전파법은 빠르다는 장점이 있다. 하지만 오차역전파를 구현하는 과정에서 버그를 발생시킬 여지가 있기 때문에 비교적 구현해내기 쉬운 수치 미분을 통해 이를 검증한다. 이처럼 두 방법을 통해 구한 기울기가 서로 같은지를 검증하는 작업을 **기울기 확인(Gradient Check)** 라고 하며, 파이썬 코드로는 아래와 같이 구현한다.

```python
# coding: utf-8
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
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
```



- 오차역전파법을 사용한 학습 구현하기

```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```



<br/>