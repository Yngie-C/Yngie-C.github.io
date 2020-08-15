---
layout: post
title: 퍼셉트론(Perceptron)
category: Deep Learning
tag: Deep-Learning
---



# Perceptron

이번 게시물에서는 모든 신경망의 기본이 되는 **퍼셉트론(Perceptron)** 에 대해서 알아보겠습니다. 신경망이 각광을 받게 된 지는 얼마되지 않았습니다만, 신경망과 퍼셉트론은 과거부터 많은 연구가 있어왔습니다. 퍼셉트론은 1957년에 고안된 알고리즘으로 다수의 신호를 입력받은 뒤 일련의 연산을 통하여 하나의 신호를 출력합니다. 아래는 단순한 퍼셉트론 하나를 이미지로 나타낸 것입니다.

![perceptron](https://missinglink.ai/wp-content/uploads/2018/11/Frame-3.png)

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



## 논리 회로

이번에는 우리의 목적이 되는 논리 회로에 대해 알아봅시다. AND, NAND, OR, XOR 게이트에 대해서 알아보겠습니다.



- **AND** 게이트

  - AND 게이트의 진리표

  |  x1  |  x2  |  y   |
  | :--: | :--: | :--: |
  |  0   |  0   |  0   |
  |  1   |  0   |  0   |
  |  0   |  1   |  0   |
  |  1   |  1   |  1   |

  이를 만족하는 퍼셉트론의 계수 $(w_1, w_2)$ 와 임곗값 $(\theta)$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(0.5, 0.5, 0.7)$ 등이 있다.

- **NAND** 게이트

  - NAND 게이트의 진리표

  |  x1  |  x2  |  y   |
  | :--: | :--: | :--: |
  |  0   |  0   |  1   |
  |  1   |  0   |  1   |
  |  0   |  1   |  1   |
  |  1   |  1   |  0   |

  이를 만족하는 퍼셉트론의 계수 $(w_1, w_2)$ 와 임곗값 $(\theta)$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(-0.5, -0.5, -0.7)$ 등이 있다.

- **OR** 게이트
  
  - OR 게이트의 진리표
  
  |  x1  |  x2  |  y   |
  | :--: | :--: | :--: |
  |  0   |  0   |  0   |
  |  1   |  0   |  1   |
  |  0   |  1   |  1   |
  |  1   |  1   |  1   |
  
  이를 만족하는 퍼셉트론의 계수 $(w_1, w_2)$ 와 임곗값 $(\theta)$ 의 예시 $(w_1, w_2, \theta)$ 로는 $(0.5, 0.5, 0.3)$ 등이 있다.

<br/>

## 3) 퍼셉트론 구현하기

- 간단한 구현부터

```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
```



- 가중치와 편향 도입 : 1) 에 등장한 퍼셉트론 수식 중 임곗값 $\theta$ 를 $-b$ 로 치환하면 아래와 같은 식이 된다.

$$
y = \begin{cases} 0 \qquad (b + w_1x_1 + w_2x_2 \leq 0) \\
1 \qquad (b + w_1x_1 + w_2x_2  > 0) \end{cases}
$$



- 가중치와 편향 구현하기 : 위의 식을 적용하여 AND 게이트를 적용하면 아래와 같다.

```python
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

아래는 NAND 게이트와 OR 게이트를 구현하는 코드이다.

```python
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```



<br/>

## 4) 퍼셉트론의 한계

- **XOR** 게이트 : XOR 게이트는 **배타적 논리합** 논리 회로이다. 두 입력 신호 중 한 쪽이 1일 때만 1을 출력한다.
  - XOR 게이트의 진리표

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  0   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  1   |  1   |  0   |

지금까지의 퍼셉트론으로는 XOR 게이트를 구현할 수 없다. AND와 OR은 선형으로 구분할 수 있는데 비해, XOR은 선형으로 출력 신호를 구분할 수 없기 때문이다. 아래 그림을 보자. 

![XOR](https://steemitimages.com/640x0/https://cdn-images-1.medium.com/max/1600/1*CyGlr8VjwtQGeNsuTUq3HA.jpeg) 

<br/>

## 5) 다층 퍼셉트론

단층 퍼셉트론으로는 XOR 게이트를 표현할 수 없지만, 여러 개의 퍼셉트론을 이어 붙이면 XOR 게이트도 표현할 수 있다. 이렇게 여러 층을 쌓아 만든 퍼셉트론을 **다층 퍼셉트론** (MultiLayer Perceptron, **MLP** )이라고 한다.

- 기존 게이트 조합하기 : 먼저 위에서 만들었던 AND, NAND, OR 게이트를 조합하여 XOR 게이트를 만들 수 있다. 아래 그림처럼 NAND 게이트와 OR 게이트를 한 층에 놓고, 그 다음 층에 AND 게이트를 배치하면 XOR 게이트와 같은 효과를 낼 수 있다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/3_gate_XOR.svg/640px-3_gate_XOR.svg.png" alt="XOR image" style="zoom: 67%;" /></p>
아래는 위와 같이 구현한 XOR게이트의 진리표이다. $s_1, s_2$ 는 각각 NAND와 OR의 출력값이다.

|  x1  |  x2  |  s1  |  s2  |  y   |
| :--: | :--: | :--: | :--: | :--: |
|  0   |  0   |  1   |  0   |  0   |
|  1   |  0   |  1   |  1   |  1   |
|  0   |  1   |  1   |  1   |  1   |
|  1   |  1   |  0   |  1   |  0   |

- XOR 게이트 구현하기 : 위에 만들었던 NAND, OR, AND를 조합하여 XOR 게이트를 구현해낼 수 있다.

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

