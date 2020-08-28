---
layout: post
title: 퍼셉트론 (Perceptron)
category: Deep Learning
tag: Deep-Learning
---



# Perceptron

이번 게시물에서는 모든 신경망의 기본이 되는 **퍼셉트론(Perceptron)** 에 대해서 알아보겠습니다. 신경망이 각광을 받게 된 지는 얼마되지 않았습니다만, 신경망과 퍼셉트론은 과거부터 많은 연구가 있어왔습니다. 퍼셉트론은 1957년에 고안된 알고리즘으로 다수의 신호를 입력받은 뒤 일련의 연산을 통하여 하나의 신호를 출력합니다. 아래는 단순한 퍼셉트론 하나를 이미지로 나타낸 것입니다.

<p align="center"><img src="https://missinglink.ai/wp-content/uploads/2018/11/Frame-3.png" alt="perceptron"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://missinglink.ai/guides/neural-network-concepts/perceptrons-and-multi-layer-perceptrons-the-artificial-neuron-at-the-core-of-deep-learning/">missinglink.ai</a></p>

위 그림에서는 총 5개의 신호 $x_1, \cdots, x_5$ 를 입력받고 있고 각 신호마다 연산을 위한 가중치 $w_1, \cdots, w_5$ 가 있습니다. 가중치는 각 신호가 주는 영향력을 조절하는 요소로 추후 이 가중치를 업데이트하여 학습을 진행해나갑니다. 모든 연산의 합이 임계값 $\theta$ 를 넘으면 $1$ 을, 넘지 못하면 $0$ 을 출력합니다. 입력 신호를 2개로 단순화하여 퍼셉트론이 작동하는 방식을 수식으로 나타내면 아래와 같습니다.

$$
y = \begin{cases} 0 \qquad (w_1x_1 + w_2x_2 \leq \theta) \\
1 \qquad (w_1x_1 + w_2x_2  > \theta) \end{cases}
$$



그리고 이를 신호가 $n$ 개인 경우로 일반화 하면 아래의 수식과 같이 나타낼 수 있습니다.


$$
y = \begin{cases} 0 \qquad (\sum^n_{i=1} w_ix_i \leq \theta) \\
1 \qquad (\sum^n_{i=1} w_ix_i > \theta) \end{cases}
$$



## Logic Gate

이번에는 우리의 목적이 되는 논리 회로에 대해 알아봅시다. AND, NAND, OR, XOR 게이트에 대해서 알아보겠습니다.

먼저 ***"AND게이트"***입니다. AND게이트는 모든 입력값이 `True`여야 `True`를 출력하고 나머지 입력에 대해서는 `False`를 출력하는 게이트입니다. 입력값 $x_1, x_2$에 대한 AND게이트의 출력값 $y$의 진리표는 다음과 같습니다.

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  0   |
|  1   |  0   |  0   |
|  0   |  1   |  0   |
|  1   |  1   |  1   |

이를 만족하는 퍼셉트론의 계수 $w_1, w_2$ 와 임곗값 $\theta$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(0.5, 0.6, 0.7)$ 등이 있습니다.

다음은 ***"NAND게이트"***입니다. *"NAND"*는 *"Not AND"*의 줄임말로 NAND게이트는 AND게이트와 같은 입력을 받아 정반대의 결과를 출력합니다. 입력값 $x_1, x_2$에 대한 NAND게이트의 출력값 $y$의 진리표는 다음과 같습니다.

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  1   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  1   |  1   |  0   |

이를 만족하는 퍼셉트론의 계수 $w_1, w_2$ 와 임곗값 $\theta$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(-0.5, -0.5, -0.7)$ 등이 있습니다.

***"OR게이트"***는 하나의 입력값만 `True`더라도 `True`를 출력하고, 모든 입력이 `False`여야 `False`를 출력하는 게이트입니다. 입력값 $x_1, x_2$에 대한 OR게이트의 출력값 $y$의 진리표는 다음과 같습니다.

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  0   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  1   |  1   |  1   |

이를 만족하는 퍼셉트론의 계수 $w_1, w_2$ 와 임곗값 $\theta$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(0.5, 0.5, 0.3)$ 등이 있다.

***"XOR 게이트"***는 <u>배타적 논리합</u>이라고도 불리는 논리 회로입니다. 두 입력 신호 중 한 쪽이 `True`일 때만 `True`를 출력합니다. 반대로 입력 신호가 같다면 `False`를 출력하게 됩니다. 입력값 $x_1, x_2$에 대한 XOR게이트의 출력값 $y$의 진리표는 다음과 같습니다.

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  0   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  1   |  1   |  0   |

이를 만족하는 퍼셉트론의 계수 $w_1, w_2$ 와 임곗값 $\theta$ 의 예시 $(w_1, w_2, \theta)$에는 어떤 것이 있을까요? 여기서 문제가 발생합니다. AND(NAND), OR 게이트의 입력값으로부터 도출되는 출력값은 선형 분류기를 사용하여 구분할 수 있었습니다. 하지만 XOR게이트의 출력값은 선형 분류기로는 구분할 수 없게 됩니다. 왜 그런지 아래 그림을 보며 알아보도록 하겠습니다.

<p align="center"><img src="https://miro.medium.com/max/700/1*CyGlr8VjwtQGeNsuTUq3HA.jpeg" alt="XOR"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://towardsdatascience.com/radial-basis-functions-neural-networks-all-we-need-to-know-9a88cc053448">towardsdatascience.com</a></p>

위 그림에서 맨 왼쪽에 있는 그림은 AND게이트의 입력값에 따른 출력값과 이를 분류하는 선형 결정 경계를 나타낸 것입니다. 가운데 있는 그림은 OR게이트의 입력값에 따른 출력값과 이를 분류하는 선형 결정 경계를 나타낸 것입니다. 맨 오른쪽은 XOR게이트의 입력값에 따른 출력값입니다. 이를 선형으로 완벽하게 분류하는 방법은 없습니다. 결정 경계를 어떻게 그리던 곡선이 들어가야 하지요.

## Multi Layer Perceptron

단층 퍼셉트론으로는 XOR게이트를 표현할 수 없습니다. 하지만 여러 개의 퍼셉트론을 이어 붙인다면 XOR게이트 아무 문제없이 구현할 수 있습니다. 이렇게 여러 층을 쌓아 만든 퍼셉트론을 **다층 퍼셉트론(Multi Layer Perceptron, MLP)**이라고 합니다.

게이트를 어떻게 쌓아올려야 XOR게이트를 구현할 수 있을까요? 아래는 NAND, OR게이트와 AND게이트를 조합하여 XOR게이트를 구현한 것입니다. 아래는 위와 같이 구현한 XOR게이트의 진리표입니다. $s_1, s_2$ 는 각각 NAND게이트와 OR게이트의 출력값이며 AND게이트는 이를 받아 값을 출력하게 됩니다.

|  x1  |  x2  |  s1  |  s2  |  y   |
| :--: | :--: | :--: | :--: | :--: |
|  0   |  0   |  1   |  0   |  0   |
|  1   |  0   |  1   |  1   |  1   |
|  0   |  1   |  1   |  1   |  1   |
|  1   |  1   |  0   |  1   |  0   |

