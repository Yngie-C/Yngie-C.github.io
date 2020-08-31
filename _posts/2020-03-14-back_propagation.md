---
layout: post
title: 오차 역전파(Back Propagation)
category: Deep Learning
tag: Deep-Learning
---







# Back Propagation

신경망은 순방향 전파로 가중치를 계산해준 뒤에 역방향 전파를 통해 더 나은 방향으로 파라미터를 조정하며 학습합니다. 이번에는 신경망 정보 전달의 핵심인 순전파와 역전파에 대해 알아보겠습니다. 아래는 신경망의 대략적인 흐름을 나타낸 이미지입니다. 이를 기억하며 아래 내용을 학습해보겠습니다.

<p align="center"><img src="https://machinelearningknowledge.ai/wp-content/uploads/2019/10/Backpropagation.gif" alt="backprop"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://machinelearningknowledge.ai/animated-explanation-of-feed-forward-neural-network-architecture/">machinelearningknowledge.ai</a></p>

## Forward Propagation

**순전파(Forward Propagation)**은 입력층에서 출력층의 방향으로 계산을 진행하는 것을 의미합니다. 아래 이미지를 보면서 순전파의 계산이 어떻게 진행되는지를 알아보겠습니다. 입력층의 노드가 2개이고 은닉층의 노드가 3개, 출력층의 노드가 1개인 신경망이 있다고 해보겠습니다. 이 인스턴스의 특성 $x_1, x_2$의 값은 1이며 레이블 $y$는 0입니다.

<p align="center"><img src="https://static.packt-cdn.com/products/9781789346640/graphics/db3e2acd-a8ad-470b-beb2-30d26a844952.png" alt="ffnn" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789346640/1/ch01lvl1sec14/feed-forward-propagation-from-scratch-in-python">packtpub.com/book/big_data_and_business_intelligence</a></p>

활성화 함수로 시그모이드 함수를 사용한다면 은닉층의 히든노드에 들어갈 값 $\text{Sigmoid}(h)$는 아래의 수식을 통해서 구할 수 있습니다. $h_1,h_2,h_3$ 은 위쪽 노드부터 활성화 함수에 들어가는 값을 나타냅니다.


$$
h_1 = 1 \times 0.8 + 1 \times 0.2 = 1.0, \quad \text{Sigmoid}(h_1) = 0.73 \\
h_2 = 1 \times 0.4 + 1 \times 0.9 = 1.3, \quad \text{Sigmoid}(h_2) = 0.79 \\
h_1 = 1 \times 0.3 + 1 \times 0.5 = 0.8, \quad \text{Sigmoid}(h_3) = 0.69
$$


이제 은닉층에 들어갈 값을 모두 구했으니 이 값으로부터 출력되는 값 $\hat{y}$를 순전파로 구해보겠습니다.


$$
\hat{y} = 0.73 \times 0.3 + 0.79 \times 0.5 + 0.69 \times 0.9 = 1.235
$$


이렇게 출력값을 구해내기 까지의 과정을 순전파라고 합니다.



## Back Propagation

위 순전파 과정에서 실제의 레이블 $y=0$ 이지만 예측 레이블은 $\hat{y} = 1.235$ 였습니다. 이렇게 다를 경우 설정한 [손실 함수(Loss function)](https://yngie-c.github.io/deep learning/2020/03/10/neural_net_training/)에 의해서 손실이 계산되고 이를 줄이기 위한 방향으로 학습이 진행됩니다. 이 때 각각의 노드에게 학습 지시를 내리는 것을 **역전파(Back propagation)**라고 합니다. 역전파에 대해 본격적으로 알아보기 전에 연쇄법칙에 대해서 먼저 알아보겠습니다.

**연쇄 법칙(Chain rule)**은 미분을 값을 전달하는 역전파를 이해하기 위해서 알아야 하는 수학적 기법 중 하나입니다. 연쇄 법칙은 합성 함수의 미분을 나타내는 방식입니다. 연쇄 법칙을 이용하면 합성 함수의 미분을 합성 함수를 구성하는 각 함수의 미분 곱으로 나타낼 수 있습니다. 예를 들어, $z$에 대한 $x,y$의 함수 $z = (x+y)^2$ 가 있다고 해보겠습니다. 이를 아래와 같이 치환할 수 있습니다.



>$$
>z = t^2, \quad t=x+y
>$$



함수 $z=f(x,y)$를 $x$에 대해 편미분한 식을 연쇄 법칙을 사용하여 다음과 같이 구할 수 있습니다.



>
>$$
>\begin{aligned}
>\frac{\partial z}{\partial x} &= \frac{\partial z}{\partial t} \frac{\partial t}{\partial x} \\
>&= 2t \cdot 1 = 2t
>\end{aligned}
>$$



이렇게 구해진 값은 편미분을 사용하여 $\frac{\partial z}{\partial x}$을 구한 결과와 같음을 볼 수 있습니다.

이제 연쇄 법칙을 통해서 역전파에 대해 알아보겠습니다. 순전파에서 신경망이 수행하는 연산을 단순화하여 위와 같이 가중치 $x,y$를 받아 출력 $z$를 내어 놓는 함수 $z = f(x,y)$ 라고 해보겠습니다. 원래의 손실 함수를 $L$이라고 하면 이를 개선하기 위한 미분은 $\frac{\partial L}{\partial z}$ 가 됩니다. 이를 각 가중치 $x, y$ 에 대하여 편미분한 값으로 나누어 가지게 됩니다. 이 부분은 순전파시 덧셈 연산을 하는 부분이므로 특별한 곱셈 연산 없이 역으로 $\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y}$ 로 나누어 가질 수 있습니다. 아래는 이 과정을 잘 보여주는 그림입니다.

<p align="center"><img src="https://bishwarup307.github.io/images/convbackprop/backprop_cs231n.png" alt="backprop_node" style="zoom:67%;" /></p>



<p align="center" style="font-size:80%">이미지 출처 : <a href="https://bishwarup307.github.io/deep%20learning/convbackprop/">bishwarup307.github.io</a></p>

위 그림에서 나타나는 과정을 신경망의 모든 노드에 대해서 실행합니다. 아래는 100번의 반복 학습을 하는 동안 각 노드의 가중치가 변화되는 과정을 보여주는 그림입니다. 그 아래는 이 과정동안 손실값의 변화와 인스턴스의 결정 경계를 시각화한 것을 각각 나타내고 있습니다.

<p align="center"><img src="https://raw.githubusercontent.com/mtoto/mtoto.github.io/master/data/2017-11-08-net/result.gif" alt="training" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://tamaszilagyi.com/blog/2017/2017-11-11-animated_net/">tamaszilagyi.com</a></p>

위 그림을 통해 손실이 감소하는 방향으로 파라미터가 개선되고 있음을 알 수 있습니다. 다음으로는 가중치가 어디서부터 어떻게 개선되는 지를 결정하는 [가중치 초기화(Weight initialization)](https://yngie-c.github.io/deep learning/2020/03/17/parameter_init/)와 [옵티마이저(Optimizer)](https://yngie-c.github.io/deep learning/2020/03/19/training_techs/)에 대해서 알아보도록 하겠습니다.

