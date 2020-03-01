---
layout: post
title: 2. 신경망(NeuralNet)
category: DL from Scratch
tag: Deep-Learning
---



## 1) 신경망

- **신경망** (Neural Net) : 퍼셉트론에서는 각자의 가중치를 직접 입력해주어야 하는 문제가 있었다. 신경망은 학습을 통해 적절한 가중치의 값을 스스로 찾아감으로서 이런 문제를 해결해준다. 신경망은 아래와 같이 생겼다. 가장 왼쪽 줄(빨간색)을 **입력층**[^1] 이라고 하고, 가장 오른쪽 줄(초록색)을 **출력층** 이라고 한다. 입력층과 출력층 사이의 층은 사람 눈에 보이지 않기 때문에 **은닉층** 이라고 한다. 

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/800px-Colored_neural_network.svg.png" alt="NeuralNet" style="zoom:40%;" />
</p>



- 이전 게시물에서 입력 신호가 2개인 퍼셉트론의 동작을 수식으로 표현했던 것이다.

$$
y = \begin{cases} 0 \qquad (b + w_1x_1 + w_2x_2 \leq 0) \\
1 \qquad (b + w_1x_1 + w_2x_2  > 0) \end{cases}
$$



- 이를 함수 $h(x)$ 를 써서 나타내면 다음과 같이 나타낼 수 있다.

$$
y = h(b + w_1x_1 + w_2x_2) \\
h(x) = \begin{cases} 0 \qquad (x \leq 0) \\
1 \qquad (x > 0) \end{cases}
$$

여기서 $h(x)$ 는 **활성화 함수** (Activation Function)라고 하며, 신호의 총합이 활성화를 일으키는지를 정하는 역할을 한다.  가중치 신호를 모두 더하여 $(\sum)$ 활성화 함수로 주입하면 새로운 값이 반환된다. 

<p align="center"><img src="https://t1.daumcdn.net/cfile/tistory/23019334560370B611" alt="activation f" style="zoom:70%;" /></p>


<br/>

## 2) 활성화 함수

- **시그모이드(Sigmoid) 함수**

  - 먼저, 수식으로 시그모이드 함수를 만나보자.

  $$
  h(x) = \frac{1}{1 + \exp(-x)}
  $$
  
  신경망에서는 활성화 함수로 위 식으로 구성된 시그모이드 함수를 이용하여 신호를 변환한다. 그리고 변환된 신호를 다음 뉴런에 전달하게 된다.



- **계단 함수 구현하기**

  - 퍼셉트론에서 구현했던 것과 같이 입력이 0을 넘으면 1을 출력하고, 그 외에는 0을 출력하는 함수를 파이썬 코드로 구현하면 다음과 같다.

  ```python
  def step_function(x):
      if x > 0:
          return 1
      else:
          return 0
  ```

  하지만 위의 코드는 $x$ 자리에 실수만 가능하고, Numpy 배열 등은 넣을 수 없다. 이런 문제를 해결하기 위해 다음과 같은 코드를 사용한다.

  ```python
  def step_function(x):
      y = x > 0
      return y.astype(np.int)
  ```

  위 함수에서는 `x > 0` 조건에 따라 `y` 에 bool, 즉 `True or False ` 값이 할당된다. 그리고 이를 int로 바꾸어 반환한다. 계단 함수의 그래프는 아래와 같다.

  <p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/1280px-Dirac_distribution_CDF.svg.png" alt="step_function" style="zoom:30%;" /></p>

- 이어서 시그모이드 함수를 파이썬 코드로 구현하면 아래와 같고, 이를 그래프로 그려보면 그 아래와 같은 결과를 얻을 수 있다. 
  
```python
  def sigmoid(x):
      return 1 / (1+np.exp(-x))
```

![sigmoid](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/480px-Logistic-curve.svg.png)



- **비선형 함수** : 계단 함수와 시그모이드 함수의 공통점은 비선형이라는 것이다. 계단 함수는 구부러진 형태로 나타나며, 시그모이드 함수는 곡선의 형태로 그려지기 때문. 신경망의 활성화 함수는 비선형의 형태여야 한다. 선형일 경우에는 신경망의 층을 깊게 하는 의미가 사라지기 때문이다. $l(x) = ax+b$ 형태의 선형 함수를 층을 쌓아 계속( $l(l(l(x)))$ 의 형태로)반복한다고 하더라도 함수의 형태는 그대로이다.



- **ReLU 함수** : Sigmoid의 단점인 [Vanishing Gradient Problem](https://brunch.co.kr/@chris-song/39) 을 해결하고자 나온 활성화 함수이다. (링크의 글에서는 Leaky ReLU 까지 다루고 있다.) ReLU 함수는 입력값이 0보다 작을 경우 0을 반환하고, 0보다 클 경우에는 입력값을 그대로 반환한다. 아래는 ReLU 함수를 수식으로 나타낸 것이고, 그 아래는 ReLU 함수를 그래프로 나타낸 것이다.
  $$
  h(x) = \begin{cases} 0 \qquad (x \leq 0) \\
  x \qquad (x > 0) \end{cases}
  $$
  
  
  ![ReLU](https://t1.daumcdn.net/cfile/tistory/2238DC3558D62AF732)
  
  다음은 ReLU 함수를 파이썬 코드로 구현한 것이다.
  
  ```python
  def relu(x):
      return np.maximum(0,x)
  ```
  
  

<br/>

## 3) 다차원 배열의 계산

- Numpy 기본에 대한 내용은 본 블로그의 [Numpy](https://yngie-c.github.io/python/2020/01/29/numpy/) 에서 확인할 수 있으며, 구글링을 통해서 더 자세하고 많은 정보를 얻을 수 있다.

<br/>

## 4) 3층 신경망 구현하기

- 표기법 설명 : 이번 장에 등장하는 가중치 표기들 중 위첨자로 표기된 $^{(k)}$ 는 $k$ 층의 가중치를 나타낸다. 그리고 아래첨자로 표기된 두개의 숫자 $_{mn}$ 은 이전 층의 $m$ 번째 뉴런에서 다음 층의 $n$ 번째 뉴런으로 전달하는 신호의 가중치임을 나타낸다. 예를 들어 $w^{(1)}_{21}$ 라는 표기가 있을 경우 1층의 가중치이며, 1층의 2번째 뉴런에서 2층의 1번째 뉴런으로 전달하는 것임을 나타낸다.



- 각 층의 신호전달 구현하기 : 가중치 신호 2개와 편향으로부터 전달받은 신호의 노드를 $a$ 라 하면

$$
a_{1}^{(1)} = w^{(1)}_{11}x_1 + w^{(1)}_{21}x_2 + b^{(1)}_{1}
$$

이고, 위 식을 행렬식으로 나타내면 다음과 같다.
$$
\mathbf{A}^{(1)} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{B}^{(1)}
$$

> $\mathbf{A}^{(1)} = (a_{1}^{(1)} \quad a_{1}^{(1)} \quad a_{1}^{(1)}), \quad \mathbf{X} = (x_1 \quad x_2), \quad \mathbf{B}^{(1)} = (b_{1}^{(1)}\quad b_{2}^{(1)} \quad b_{3}^{(1)})$ 
>
> $\mathbf{W}^{(1)} = \begin{pmatrix} w^{(1)}_{11} w^{(1)}_{21} w^{(1)}_{31} \\ w^{(1)}_{12} w^{(1)}_{22} w^{(1)}_{32} \end{pmatrix}$

위의 행렬식을 구현하는 파이썬 코드는 다음과 같다. 각 신호와 가중치의 값은 책에 있는 것을 그대로 사용하였다. 구해진 값에 활성화 함수(여기서는 시그모이드)를 취해주면 새로운 층의 신호행렬 $Z^{(1)}$ 을 얻을 수 있다.

```python
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)
```

은닉층에서는 위와 같은 과정을 반복하여 신호를 전달하며, 마지막 출력층에서만 분류나 회귀에 맞는 활성화 함수를 사용한다.



<br/>

## 5) 출력층 설계하기

대개 출력층에서의 활성화 함수로 회귀에서는 (자기 자신을 그대로 출력하는) 항등 함수를, (다중 레이블) 분류에서는 소프트맥스 함수를 사용한다.   

- **소프트맥스(softmax)** 함수 : 소프트맥스 함수를 수식으로 표현하면 아래와 같고, 그 아래는 소프트맥스 함수를 파이썬 코드로 구현한 것이다.

$$
y_k = \frac{\exp(a_k)}{\sum^n_{i=1}\exp(a_i)}
$$



```python
def softmax(a):
	exp_a = np.exp(a)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
    
    return y
```



- 소프트맥스 함수 구현 시 주의점 : 소프트맥스 함수가 지수함수이기 때문에 $a$ 값에 따라서 결과값이 매우 커지는 오버플로(overflow) 문제가 발생한다. 컴퓨터는 큰 값으로 나눌 때, 수치가 불안정해지므로 이런 오버플로 문제를 해결해주어야 한다. 아래는 이를 해결하기 위한 수식이다.

$$
y_k = \frac{\exp(a_k)}{\sum^n_{i=1}\exp(a_k)} = \frac{C\exp(a_k)}{C\sum^n_{i=1}\exp(a_k)} \\ = \frac{\exp(a_k +\log C)}{\sum^n_{i=1}\exp(a_k + \log C)} = \frac{\exp(a_k +C^\prime)}{\sum^n_{i=1}\exp(a_k + C^\prime)}
$$

> 여기서 $C^\prime$ 은 어떤 값이든 상관 없지만 오버플로를 막기 위해서는 일반적으로 입력 신호 중 최댓값(절댓값이 가장 큰 것)을 입력한다.

위의 수식을 파이썬 코드로 옮기면 다음과 같이 쓸 수 있다.

```python
def softmax(a):
    c = np.max(a)
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
    
    return y
```

소프트맥스 함수의 출력값을 모두 더하면 1이 되며, 이 때문에 출력값을 확률로도 해석할 수 있다.



- 출력층의 뉴런 수 정하기 : 출력층의 뉴런 수는 무넺에 맞게 적절히 정해야 한다.



[^1]: 입력층의 경우 따로 명시하지 않는다는 의견도 있다. 