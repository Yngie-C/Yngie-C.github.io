---
layout: post
title: 손실 함수 (Loss Function)와 경사 하강법 (Gradient Descent)
category: Deep Learning
tag: Deep-Learning
---



# Learning

신경망은 파라미터를 더 좋은 방향으로 개선하면서 학습을 진행해나가게 됩니다. 그런데 '더 좋은 방향'이란 것은 무엇일까요? 우리가 길을 갈 때에도 어느 쪽 방향으로 갈지를 정하려면 목적지나 기준이 있어야 합니다. 이런 기준이 되는 손실 함수에 대해서 알아보겠습니다.

## Loss function

신경망 노드 내의 파라미터가 어느 방향으로 개선되어야 할 지 판단하는 지표로는 **손실(Loss, Cost)**을 사용합니다. 어떤 작업을 수행할 지에 따라 이 손실을 구하기 위한 **손실 함수(Loss function, Cost function)**이 달라집니다.

### MSE

회귀(Regression)의 손실 함수로는 일반적으로 **평균 제곱 오차(Mean square error, MSE)**를 사용합니다. 평균 제곱 오차는 연속형으로 나타나는 실제 레이블과 예측 레이블의 차이에 제곱해준 값을 모든 인스턴스에 대해 합을 구한 것입니다. 수식으로는 아래와 같이 나타낼 수 있습니다.


$$
\text{Loss}_{MSE} = \frac{1}{2}\sum_k{(\hat{y}_k - y_k)^2}
$$


위 식에서 $\hat{y}_k$는 신경망의 출력값이며 $y_k$는 실제 레이블입니다. $k$는 데이터의 개수입니다. 아래는 각 인스턴스에 대해 MSE 오차를 시각화하여 나타낸 것입니다.

<p align="center"><img src="https://cdn-media-1.freecodecamp.org/images/MNskFmGPKuQfMLdmpkT-X7-8w2cJXulP3683" alt="mse" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.freecodecamp.org/news/machine-learning-mean-squared-error-regression-line-c7dde9a26b93/">freecodecamp.org</a></p>

MSE는 직관적이고 이해하기 쉬우면서도 식을 미분했을 때 나오는 식의 계산이 용이하므로 회귀에서 주로 사용되는 손실 함수입니다.

### Cross-Entropy Error

분류(Classification)에서는 교차 엔트로피 오차(Cross-entropy error, CEE)를 주로 사용합니다. 교차 엔트로피 오차를 사용하여 구한 오차를 수식으로 나타내면 아래와 같습니다.


$$
\text{Loss}_{CEE} = -\sum_k{y_k\log{\hat{y}_k}}
$$

교차 엔트로피 오차는 어떻게 작용할까요? 실제 레이블이 $\left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 0\end{array}\right]$ 인 5개의 인스턴스에 대하여 3개의 분류기가 각각 다음과 같이 예측값을 내놓았다고 가정하겠습니다.


$$
\text{Classifier 1} : \left[\begin{array}{ccccc}1 & 0 & 0 & 0 & 0\end{array}\right]\\
\text{Classifier 2} : \left[\begin{array}{ccccc}0 & 0 & 0 & 0 & 0\end{array}\right]
$$
이 때의 교차 엔트로피 오차를 구해보겠습니다. 실제로 $\log0 = -\infty$ 이므로 $0$ 대신 매우 작은 값인 $2^{-10}$을 대입하여 $\log 2^{-10^{10}} = -10^{10}$으로 대체하겠습니다. (실제로는 이 값보다 훨씬 더 작은 값을 나타냅니다)
$$
\begin{aligned}
L_1 &= -(1\log1 + 0\cdot \log 2^{-10^{10}} + 0\cdot \log 2^{-10^{10}} + 0\cdot \log 2^{-10^{10}} + 0\cdot \log 2^{-10^{10}}) \\
&= - (1\cdot 0 + 0\cdot -10^{10} +0\cdot -10^{10}+0\cdot -10^{10}+0\cdot -10^{10})= 0 \\
L_2 &= -(1\cdot \log 2^{-10^{10}} + 0\cdot \log 2^{-10^{10}} + 0\cdot \log 2^{-10^{10}} + 0\cdot \log 2^{-10^{10}} + 0\cdot \log 2^{-10^{10}}) \\
& = - (1\cdot -10^{10} + 0\cdot -10^{10} +0\cdot -10^{10}+0\cdot -10^{10}+0\cdot -10^{10})= 10^{10}
\end{aligned}
$$
정답 레이블을 모두 맞춘 첫 번째 분류기의 교차 엔트로피 오차는 0이지만, 모두 맞추지 못한 두 번째 분류기의 교차 엔트로피 오차는 매우 큰 값이 나오는 것을 볼 수 있습니다. $0$으로 갈 때 급격하게 값이 $-\infty$로 발산하는 로그 함수의 특성을 사용하여 적절한 손실값이 나오도록 할 수 있습니다. 아래는 로그 함수의 그래프입니다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Binary_logarithm_plot_with_ticks.svg/1280px-Binary_logarithm_plot_with_ticks.svg.png" alt="log_func" style="zoom: 33%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/Logarithm">wikipedia - Logarithm</a></p>

## Gradient Descent

기준이 되는 손실 함수에 대한 내용을 익혔습니다. 손실 함수는 말 그대로 손실(Loss)로써 손실이 높다는 것은 신경망 모델의 성능이 좋지 않음을 나타내는 지표가 됩니다. 그렇기 때문에 우리는 모델의 손실을 줄이는 방향으로 학습해가야 합니다. 신경망에서는 파라미터를 개선하는 방법으로 **경사 하강법(Gradient descent method)**을 사용합니다.

### Gradient

경사 하강법에서 중요한 것은 당연히 경사(혹은 기울기)입니다. 함수의 기울기는 미분을 사용하여 구합니다. 아래 그림

임의의 미분가능한 함수 $f(x)$에 대한 미분은 다음과 같은 수식을 사용하여 구할 수 있습니다.


$$
\frac{df(x)}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}
$$


편수가 여러 개인 경우에는 편미분(Partial derivative)을 사용합니다. 특정 변수 하나에 초점을 맞춘 뒤에 다른 모든 변수는 상수로 취급하여 변수가 1개일 때의 미분과 같은 방법을 적용하는 것을 편미분이라고 합니다. 간단한 예를 들어보겠습니다. 예를 들어, $g(x, y, z) = 3 x^2 y + x y^2 z + 2x + yz$ 를 $x$ 에 대해 편미분하면 다음과 같은 식이 도출됩니다.


$$
\frac{\partial g(x)}{\partial x} = 6xy + y^2z + 2
$$


$x$에 대한 편미분이므로 $y, z$ 모두를 $3, 2$ 와 같은 상수로 취급하여 쉽게 미분할 수 있습니다. 

다음은 **그래디언트(Gradient)**에 대해 알아보겠습니다. 변수가 여러 개일 때 모든 변수에 대한 편미분을 하나로의 벡터로 묶어 정리한 것을 그래디언트라고 합니다. 그래디언트도 간단한 예시를 통해 알아보겠습니다. $h(x_1, x_2, \cdots, x_n)$ 과 같이 변수가 $n$ 개인 함수의 그래디언트는 다음과 같이 나타나게 됩니다.


$$
\nabla_x h(x) = \bigg[\frac{\partial h(x)}{\partial x_1}, \frac{\partial h(x)}{\partial x_2}, \cdots, \frac{\partial h(x)}{\partial x_n}\bigg]
$$


### Descent

우리가 다루는 손실 함수는 모두 아래로 볼록(Convex, 컨벡스)함수 입니다. 아래로 볼록 함수가 어떻게 생겼는지는 아래의 그림을 통해서 살펴보겠습니다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/ConvexFunction.svg/1920px-ConvexFunction.svg.png" alt="convex" style="zoom:30%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/Convex_function">wikipedia - convex function</a></p>

2차원에서 컨벡스 함수란 함수 $f(x)$ 위의 두 점 $(x_1, f(x_1)), (x_2, f(x_2))$ 을 잇는 직선 $g(x)$ 에 대하여 구간 $(x_1, x_2)$ 사이의 점 $x_m$ 에서 $f(x) < g(x)$ 를 만족하는 함수입니다. 이를 3차원으로 확장하면 다음과 같은 함수가 그려집니다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/6/6e/Grafico_3d_x2%2Bxy%2By2.png" alt="convex_3d" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/Convex_function">wikipedia - convex function</a></p>

우리의 목적은 대략 위처럼 생긴 손실 함수의 최저점을 구하는 것입니다. 이런 경우에는 내리막길인 방향으로만 가는 방법, 즉 경사 하강법을 적용하여 최저점인 지점을 찾을 수 있습니다. 저렇게 생긴 함수의 어떤 점에서 시작하더라도 내리막길인 방향으로만 발걸음을 내딛는다면 최저점(Minimum)에 도착할 수 있게될 것입니다. (물론 특성이 많은 경우 완벽한 컨벡스 함수가 아니기 때문에 지역적 최저점(Local minimum)에 빠질 수 있다는 문제점이 있지만 그래도 최저인 점을 찾기 위해서는 그래디언트가 감소하는 방향으로 나아가야 합니다.)

변수 $x_i$ 에 대한 경사 하강법을 수식으로 나타내면 다음과 같이 나타낼 수 있습니다.


$$
x_i = x_i - \eta \frac{\partial f(x)}{\partial x_i}
$$


위 식에서 $\eta$ 는 학습률(Learning rate)이며 한 번의 학습으로 얼마나 파라미터 값을 갱신할 지를 결정하는 중요한 파라미터입니다. 학습률이 너무 작으면 최저점을 찾아가는데 너무 많은 시간이 소요되며, 심한 경우에는 주어진 반복수 내에 최저점을 찾지 못하는 경우도 발생하게 됩니다. 반대로 학습률을 너무 크게 설정하게 되면 최저점을 그냥 지나쳐버리기 때문에 제대로된 학습을 하지 못하고 발산해버리는 문제가 발생합니다. 적절한 학습률을 찾는 것은 신경망을 최적화하는 데에 있어 중요한 문제이기 때문에 학습률 감쇠(Learning rate decay) 등 다양한 전략을 사용합니다. 경사 하강법에 대한 더욱 자세한 정보는 [이곳](https://yngie-c.github.io/machine learning/2020/04/07/Linear_reg/)에서 볼 수 있습니다.