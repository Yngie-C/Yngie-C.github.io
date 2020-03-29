---
layout: post
title: 3. 신경망 학습
category: Deep Learning
tag: Deep-Learning
---



## 1) 데이터에서 학습하기

- 데이터 주도 학습 : 
- 훈련 데이터와 실험 데이터 : 



<br/>

## 2) 손실 함수

모델의 지표를 판단하기 위한 지표로 **손실** (Loss, Cost)을 사용하며 이를 구하기 위한 지표를 **손실 함수** (Loss Function, Cost Function)라고 한다.

- **평균 제곱 오차(MSE)** : 대표적인 손실 함수로 평균 제곱 오차(Mean Square Error, MSE)가 있다. 평균 제곱 오차를 수식으로 나타내면 아래와 같다.

$$
E = \frac{1}{2}\sum_k{(\hat{y}_k - y_k)^2} \\
\hat{y}_k\text{ : 신경망의 출력값} \quad y_k\text{ : 실제 레이블} \quad k\text{ : 데이터의 차원 수}
$$

평균 제곱 오차를 파이썬 코드로 구현한 것은 다음과 같다.

```python
def mean_square_error(y, t):
    return 0.5 * np.sum((y - t)**2)
```



- **교차 엔트로피 오차** : 다른 대표적인 손실 함수로 교차 엔트로피 오차(Cross-Entropy Error, CEE)도 있다. 교차 엔트로피 오차를 수식으로 나타내면 아래와 같다.

$$
E = -\sum_k{y_k\log{\hat{y}_k}}
$$

원-핫 인코딩의 경우를 생각해보자. 5개의 클래스가 있는 레이블에서, 실제 레이블 $t_k$ 이 [1, 0, 0, 0, 0] 이면 예측 레이블이 [1, 0, 0, 0, 0]이 되는 경우를 제외하고는 교차 엔트로피 오차가 매우 커지게 된다. $-\log{x}$ 함수는 $x$ 가 0에 가까워질수록 함수값이 기하급수적으로 커지기 때문이다. 교차 엔트로피 오차를 파이썬 코드로 구현한 것은 다음과 같다.

```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))
# 여기서 delta는 로그 함수에 0이 들어가서 에러가 나는 것을 방지하기 위한 아주 작은 임의의 값이다.
```



- **미니배치 학습** : 전체 데이터는 너무 크기 때문에, 그 중 일부 데이터를 추려내고 그 손실들의 합을 계산한다. 이 과정에서 뽑아낸 일부를 미니배치라고 부른다. 미니배치는 `np.random.choice` 함수를 사용하여 구현할 수 있다. 아래는 미니배치를 추려내는 코드이다.

```python
train_size = x_train.shape[0]
batch_size = 10		# 배치 사이즈의 크기이다. 32, 64등을 주로 사용한다.
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]
```



- **(배치용) 교차 엔트로피 오차 구현하기** : 교차 엔트로피 오차를 구현하는 파이썬 코드는 다음과 같다.

```python
def cross_entropy_error(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = t.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7)) / batch_size
```



- **손실 함수를 사용하는 이유** : 정확도라는 Metric을 두고 굳이 손실 함수를 사용하는 이유는 바로 **미분** 때문이다. 정확도를 사용하게 되면 대부분의 지점에서 미분 값이 0이 되기 때문에 매개 변수를 갱신하기가 어렵다. 예를 들어 100개 데이터의 이진 분류 문제를 생각해보자. 가능한 정확도는 0%, 1%, 2%, ... 49%, 50%, 51%, ... 98%, 99%, 100% 처럼 불연속적인 값으로 주어진다. 이를 함수로 나타내면 (100개의 계단이 있는) **계단 함수 **가 된다. 계단 함수는 대부분의 지점에서 미분값이 0이므로 적절한 매개변수를 찾아 파라미터를 개선해 나갈 수 없게 된다.




<br/>

## 3) 수치 미분

- **미분** : 한 순간의 변화량을 표현한 것을 미분이라 하며, 이것을 수식으로 나타내면 다음과 같다.

$$
\frac{df(x)}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}
$$

파이썬 코드로 위 식을 구현하면 아래와 같다.

```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x))/h
```

위 코드에서 $h$ 의 값은 작으면 작을수록 좋지만, 너무 작게 되면 컴퓨터의 특성상 반올림 오차가 발생하게 되어 컴퓨터가 $h$ 값을 0으로 인식하게 된다. 위 코드에서처럼 $10^-4$ 를 사용하면 좋은 결과를 얻는다고 알려져 있지만 이로 인해 일정 정도의 오차가 발생하게 된다. 이를 줄이기 위해 중앙 차분을 사용하여 `return` 값으로 `(f(x+h)-f(x-h))/2h`  를 사용하기도 한다.



- **편미분** : 변수가 여러 개인 경우에는 편미분을 사용한다. 특정 변수 하나에 초점을 맞춘 뒤, 다른 모든 변수를 상수로 취급하여 변수가 1개일 때 미분과 같은 방법을 적용하는 것을 편미분이라고 한다.



<br/>

## 4) 그라디언트

- **그라디언트(Gradient)** : 모든 변수에 대한 편미분을 하나로 묶어 벡터로 정리한 것을 그라디언트라고 한다. 그라디언트를 구하기 위한 파이썬 코드는 다음과 같다.

```python
def numerical_diff(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad
```



- **경사 하강법(Gradient Descent)** : 최적의 모델은 손실 함수가 최저가 될 **때** 의 모델. 그 **때** 를 찾기 위해서 그라디언트를 사용하게 된다. 기울어진 방향(그라디언트가 0이 되는 방향)이 꼭 최솟값을 나타내는 것은 아니다.[^1] 하지만 그 방향으로 가야 더 작은 손실(을 보이는 모델)을 얻을 수 있는 것은 분명하기 때문에 그라디언트를 통해 모델을 개선해 나갈 방향을 잡는다. 경사 하강법을 수식으로 나타내면 다음과 같다.

$$
x_i = x_i - \eta \frac{\partial{f}}{\partial{x_i}} \\
\eta : \text{학습률(learning rate)}
$$

학습률은 한 번의 학습으로 얼마만큼 학습해야 할 지를 결정하는 것이 학습률이다. 학습률이 너무 작으면 손실이 가장 적은 점을 찾아가는데 너무 많은 시간이 소요되며, 너무 크면 최저점을 그냥 지나쳐버려 제대로된 학습을 할 수 없다. 때문에 적절한 학습률을 찾는 것은 딥러닝에 있어 중요한 문제이다.

경사 하강법은 파이썬 코드로 다음과 같이 구현할 수 있다.

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
```



- 신경망에서의 그라디언트 : 신경망에서도 그라디언트를 구해야 한다. `shape` 이 2X3이고, 가중치가 $\mathbf{W}$ , 손실 함수가 $L$ 인 신경망을 예로 들어보자. 이 때의 그라디언트를 수식으로 나타내면 다음과 같다,

$$
\mathbf{W} = \left(\begin{array}{ccc} w_{11} w_{12} w_{13} \\ w_{21} w_{22} w_{23} \end{array} \right) \\ \quad \\
\frac{\partial{L}}{\partial{\mathbf{W}}} = \left(\begin{array}{ccc} \frac{\partial{L}}{\partial{w_{11}}} \frac{\partial{L}}{\partial{w_{12}}} \frac{\partial{L}}{\partial{w_{13}}} \\ \frac{\partial{L}}{\partial{w_{21}}} \frac{\partial{L}}{\partial{w_{22}}} \frac{\partial{L}}{\partial{w_{23}}} \end{array} \right)
$$

아래는 간단한 신경망(SimpleNet class)로 기울기를 구하는 코드이다. 

```python
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x) 
        y = softmax(z)	# 이전에 정의했던 softmax함수를 사용
        loss = cross_entropy_error(y, t)

        return loss
```



<br/>

## 5) 학습 알고리즘 구현하기

- 신경망 학습의 절차

  - 1단계 : 훈련 데이터 중 일부를 무작위로 가져온다. 이 일부의 데이터를 미니배치라고 하며, 그 미니배치의 손실함수 값을 줄이는 것이 목표.
  - 2단계 : 손실을 줄이기 위해 각 가중치 매개변수의 그라디언트를 구한다. 그라디언트는 매개변수를 개선해나갈 방향을 제시한다.
  - 3단계 : 가중치 매개변수를 기울기 방향으로 갱신하며, 갱신하는 정도는 학습률로 조정한다.
  - 4단계 : 1~3 단계의 반복

  이 과정에서 미니배치를 무작위로 선정하기 때문에 확률적 경사 하강법(Stochastic Gradient Descent, SGD)라고 부른다.



- 2층 신경망 클래스 구현하기 : 처음에는 2층으로 구성된 신경망을 하나의 클래스로 구현하는 것부터 시작한다. 이 클래스를 구현하는 코드는 아래와 같다.

```python
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        #앞에서 살펴본 cross_entropy_error 함수를 통해 손실 값을 구한다.
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
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
    
    #numerical_gradient(수치 미분)가 아니라 오차역전파법을 통한 gradient함수. 수치 미분과 비교하여 훨씬 더 빠른 시간 내에 그라디언트 값을 구할 수 있다.
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        # params['W1'], params['W2']각각 1, 2번째 층의 가중치
        b1, b2 = self.params['b1'], self.params['b2']
        # params['b1'], params['b2']각각 1, 2번째 층의 편향
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
		# grads['W1'], grads['W2']각각 1, 2번째 층의 가중치 기울기
        # grads['b1'], grads['b2']각각 1, 2번째 층의 편향 기울기
        return grads
```



- 미니배치 학습 구현하기 : 위에서 만든 클래스를 빌려 미니배치를 활용한 MNIST 손글씨 데이터의 학습을 구현해볼 수 있다.

```python
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
```



- 시험 데이터로 평가하기 : 그래프에서 손실 함수의 값이 점점 내려가는 것을 볼 수 있지만, 이는 학습 데이터에 대한 손실일 뿐이지 새로운 데이터에 대해서 이 모델이 잘 작동할지는 알 수 없다. 그렇기 때문에 에포크마다 시험 데이터와 훈련 데이터의 정확도를 기록하는 것이 좋다. 위 학습 코드에 아래 코드 중 활성화된 부분을 맞게 추가 입력하면, 훈련 데이터와 시험 데이터에 대한 정확도를 얻을 수 있다. 

```python
"""
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []"""
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

"""
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)"""
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
```

위 모델을 통해 얻어진 훈련 데이터와 시각 데이터에 대한 정확도를 시각화하는 코드는 아래와 같이 쓸 수 있다.

```python
# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```



<br/>