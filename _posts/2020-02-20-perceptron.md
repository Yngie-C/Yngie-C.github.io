---
layout: post
title: 퍼셉트론 (Perceptron)
category: Deep Learning
tag: Deep-Learning
---



# Perceptron

이번 게시물에서는 모든 신경망(Neural net)의 기본이 되는 **퍼셉트론(Perceptron)** 에 대해서 알아보겠습니다. 신경망이 각광을 받게 된 지는 얼마되지 않았습니다만, 그보다 훨씬 전부터 신경망과 퍼셉트론에 대해서 많은 연구가 있어왔습니다. 퍼셉트론은 1957년에 고안된 알고리즘으로 다수의 신호를 입력받은 뒤 일련의 연산을 통하여 하나의 신호를 출력합니다. 아래는 단순한 퍼셉트론 하나를 이미지로 나타낸 것입니다.

<p align="center"><img src="https://missinglink.ai/wp-content/uploads/2018/11/Frame-3.png" alt="perceptron"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://missinglink.ai/guides/neural-network-concepts/perceptrons-and-multi-layer-perceptrons-the-artificial-neuron-at-the-core-of-deep-learning/">missinglink.ai</a></p>

위 퍼셉트론은 총 5개의 신호 $(x_1, \cdots, x_5)$ 를 입력받습니다. 각 신호는 연산을 위한 가중치 $(w_1, \cdots, w_5)$ 를 가지고 있습니다. 가중치는 각 신호가 주는 영향력을 조절하는 요소로 추후 학습 과정에서 이 값을 업데이트하게 됩니다. 퍼셉트론은 모든 연산의 합이 임계값 $\theta$ 를 넘으면 $1$ 을, 넘지 못하면 $0$ 을 출력합니다. 입력 신호를 2개로 단순화하여 퍼셉트론이 작동하는 방식을 수식으로 나타내면 아래와 같습니다.

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

이번에는 **논리 회로(Logic gate)**에 대해 알아보겠습니다.

처음으로 알아볼 게이트는 ***"AND게이트"***입니다. AND게이트는 모든 입력값이 `True`일 때만 `True`를 출력하는 게이트입니다. 나머지 입력에 대해서는 `False`를 출력합니다. 입력 신호 $(x_1, x_2)$ 에 대한 AND게이트의 출력값 $y$ 의 진리표는 다음과 같습니다.

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  0   |
|  1   |  0   |  0   |
|  0   |  1   |  0   |
|  1   |  1   |  1   |

이를 만족하는 퍼셉트론의 계수 $w_1, w_2$ 와 임곗값 $\theta$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(0.5, 0.6, 0.7)$ 등이 있습니다. AND 게이트는 순서도 상에서 아래와 같이 나타낼 수 있습니다.

![and_gate](https://pxt.azureedge.net/blob/21beeee46f4bf464af0ed4dfe895b7b5670a1b41/static/courses/logic-lab/logic-gates/and-gate.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://makecode.microbit.org/courses/logic-lab/logic-gates">makecode.microbit.org</a></p>

다음은 ***"NAND게이트"***입니다. *"NAND"*는 *"Not AND"*의 줄임말로 NAND게이트는 AND게이트와 같은 입력을 받아 정반대의 결과를 출력합니다. 입력값 $(x_1, x_2)$ 에 대한 NAND게이트의 출력값 $y$ 의 진리표는 아래와 같습니다.

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  1   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  1   |  1   |  0   |

이를 만족하는 퍼셉트론의 계수 $w_1, w_2$ 와 임곗값 $\theta$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(-0.5, -0.5, -0.7)$ 등이 있습니다. NAND 게이트는 순서도 상에서 아래와 같이 나타낼 수 있습니다.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Nand-gate-en.svg/1280px-Nand-gate-en.svg.png" alt="nand_gate" style="zoom: 33%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://commons.wikimedia.org/wiki/File:Nand-gate-en.svg">commons.wikimedia.org</a></p>

***"OR게이트"***는 하나의 입력값만 `True`이면 `True`를 출력하는 게이트입니다. 즉, 모든 입력이 `False`여야 `False`를 출력하게 됩니다. 입력값 $(x_1, x_2)$에 대한 OR게이트의 출력값 $y$ 의 진리표는 다음과 같습니다.

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  0   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  1   |  1   |  1   |

이를 만족하는 퍼셉트론의 계수 $w_1, w_2$ 와 임곗값 $\theta$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(0.5, 0.5, 0.3)$ 등이 있다. OR 게이트는 순서도 상에서 아래와 같이 나타낼 수 있습니다.

![or_gate](https://pxt.azureedge.net/blob/a19b61032bbf27fb26135da3512fa5349f69c6d5/static/courses/logic-lab/logic-gates/or-gate.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://makecode.microbit.org/courses/logic-lab/logic-gates">makecode.microbit.org</a></p>

***"XOR 게이트"***는 <u>배타적 논리합</u>이라고도 불리는 논리 회로입니다. 두 입력 신호 중 한 쪽이 `True`일 때만 `True`를 출력합니다. 반대로 입력 신호가 같다면 `False`를 출력하게 됩니다. 입력값 $x_1, x_2$에 대한 XOR게이트의 출력값 $y$의 진리표는 다음과 같습니다.

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  0   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  1   |  1   |  0   |

이를 만족하는 퍼셉트론의 계수 $w_1, w_2$ 와 임곗값 $\theta$ 의 예시 $(w_1, w_2, \theta)$에는 어떤 것이 있을까요? 언뜻 생각해봐도 쉽게 떠올리기 쉽지 않습니다.

당연합니다. 2개의 신호를 받는 퍼셉트론 하나로는 XOR 게이트를 만들 수 없기 때문이지요. 이유를 아래 이미지에서 보도록 하겠습니다. 

<p align="center"><img src="https://miro.medium.com/max/700/1*CyGlr8VjwtQGeNsuTUq3HA.jpeg" alt="XOR"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://towardsdatascience.com/radial-basis-functions-neural-networks-all-we-need-to-know-9a88cc053448">towardsdatascience.com</a></p>

2차원 상에서 신호 2개 $(x_1, x_2)$와 가중치 2개 $(w_1, w_2)$와 임계값 $\theta$ 로 만들어지는 경계면은 직선입니다. 위 그림에서 알 수 있듯 AND(NAND) 게이트와 OR 게이트의 출력값은 선형 분류기를 사용하여 구분할 수 있습니다. AND 게이트에서는 $(1,0), (0,1), (1,1)$ 사이의 두 점을 지나는 직선을, OR 게이트에서는 $(0,0), (0,1), (1,0)$ 사이의 두 점을 지나는 직선을 구하면 되는 것이지요. 하지만 아무리 생각해봐도 가장 오른쪽에 있는 XOR 게이트의 출력값을 분류할 수 있는 직선은 없습니다. 그래서 XOR 게이트의 $(w_1, w_2, \theta)$ 를 쉽게 떠올리기 쉽지 않은 것이지요.

## Multi Layer Perceptron

즉, 단층 퍼셉트론으로는 XOR 게이트를 표현할 수 없습니다. 그렇다면 XOR 게이트는 어떻게 구현할 수 있을까요? 한 개가 안된다면 두 개를 쌓으면 됩니다. 두 개의 퍼셉트론을 이어 붙이면 XOR게이트를 아무 문제없이 구현할 수 있습니다. 이렇게 2개 이상의 층을 쌓아 만든 퍼셉트론을 **다층 퍼셉트론(Multi Layer Perceptron, MLP)**이라고 합니다.

게이트를 어떻게 쌓아올려야 XOR게이트를 구현할 수 있을까요? 아래는 NAND, OR 게이트와 AND 게이트를 조합하여 XOR게이트를 구현한 것입니다. 아래는 위와 같이 구현한 XOR게이트의 진리표입니다. $s_1, s_2$ 는 각각 NAND게이트와 OR게이트의 출력값이며 AND게이트는 이를 받아 값을 출력하게 됩니다.

|  x1  |  x2  |  s1  |  s2  |  y   |
| :--: | :--: | :--: | :--: | :--: |
|  0   |  0   |  1   |  0   |  0   |
|  1   |  0   |  1   |  1   |  1   |
|  0   |  1   |  1   |  1   |  1   |
|  1   |  1   |  0   |  1   |  0   |

이렇게 구현한 XOR 게이트의 순서도는 다음과 같습니다. 먼저, 동일한 신호가 NAND 게이트와 OR 게이트에 입력됩니다. 그리고 각 게이트에서의 출력값이 다시 AND 게이트의 입력값으로 들어가게 되지요. 이렇게 출력된 AND 게이트의 출력값 $y$ 과 맨 처음에 입력된 신호 $(x_1, x_2)$ 를 비교해보면 XOR 게이트로 작동하고 있음을 알 수 있습니다. 

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/3_gate_XOR.svg/1920px-3_gate_XOR.svg.png" alt="xor_gate" style="zoom: 25%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/XOR_gate">en.wikipedia.org</a></p>

XOR 게이트는 순서도에서 다음과 같이 단순화하여 나타냅니다.

![xor_gate2](https://pxt.azureedge.net/blob/3c3cfeb0235eead736bc0d90d23de5f6e74c1abe/static/courses/logic-lab/logic-gates/xor-gate.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://makecode.microbit.org/courses/logic-lab/logic-gates">makecode.microbit.org</a></p>