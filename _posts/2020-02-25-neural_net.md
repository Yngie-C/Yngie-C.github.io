---
1layout: post
title: 신경망 (Neural Network)
category: Deep Learning
tag: Deep-Learning
---



# Neural Network

퍼셉트론(Perceptron)에서는 게이트를 만들어 주기 위해서 게이트 마다의 가중치를 직접 입력해주어야 하는 어려움이 있었습니다. **신경망(Neural Net)**은 수동으로 파라미터를 입력해주지 않아도 알아서 파라미터를 결정할 수 있도록 하는 장치입니다.

신경망의 구조는 아래 그림과 같습니다.

<p align="center"><img src="https://i.imgur.com/McMOhuQ.png" alt="NeuralNet" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.kdnuggets.com/2017/10/neural-network-foundations-explained-gradient-descent.html">kdnuggets.com</a></p>

퍼셉트론의 수식을 다시 떠올려 보겠습니다. 아래는 2개의 입력값 $x_1, x_2$을 받는 퍼셉트론을 수식으로 나타낸 것입니다. $w_1, w_2$는 각 입력값에 곱해지는 가중치이며 $b$는 편향(Bias)입니다.


$$
y = \begin{cases} 0 \qquad (b + w_1x_1 + w_2x_2 \leq 0) \\
1 \qquad (b + w_1x_1 + w_2x_2  > 0) \end{cases}
$$



이를 함수 $h(x)$를 사용하면 다음과 같이 나타낼 수 있습니다. 
$$
y = h(b + w_1x_1 + w_2x_2) \\
h(x) = \begin{cases} 0 \qquad (x \leq 0) \\
1 \qquad (x > 0) \end{cases}
$$

여기서 $h(x)$는 **활성화 함수(Activation function)**라고 하며, 신호의 총합이 활성화를 일으키는지를 정하는 역할을 합니다. 가중치 신호를 모두 더하여 활성화 함수로 주입하면 임계값과 비교하여 출력값이 반환됩니다. 아래는 이 과정을 나타낸 것입니다. 

<p align="center"><img src="https://www.i2tutorials.com/wp-content/media/2019/09/Deep-learning-20-i2tutorials.png" alt="activation f" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.i2tutorials.com/explain-activation-function-in-neural-network-and-its-types/">i2tutorials.com</a></p>

## Activation Function

###  Step Function

활성화 함수는 신경망의 행동을 결정하는 중요한 역할을 합니다. 위에서 나타낸 것처럼 임계값을 기준으로 출력값이 $0-1$ 로 변하는 함수를 **계단 함수(Step function)**라고 합니다. 계단 함수의 그래프는 다음과 같이 생겼습니다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/1280px-Dirac_distribution_CDF.svg.png" alt="step_function" style="zoom: 33%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/Heaviside_step_function">wikipedia - Heaviside step function</a></p>

계단 함수는 직관적으로 이해할 수 있는 활성화 함수지만 불연속이라는 단점이 있습니다. 신경망에서는 학습을 위해 미분을 해야하는데 계단 함수는 임계값 지점에서 미분이 불가능할 뿐더러, 다른 지점에서는 미분값이 $0$이 되어 학습을 할 수 없게 됩니다. 이런 한계점을 해결하기 위해서 등장한 것이 **시그모이드 함수(Sigmoid)**입니다.

### Sigmoid Function

시그모이드 함수는 기본적으로 $S$ 모양을 그리는 곡선 함수를 통칭하여 부르는 말이며 대표적인 함수는 로지스틱(Logistic) 함수와 하이퍼탄젠트(Hyper tangent, $\tanh$) 함수가 있습니다. 두 함수의 그래프를 보며 시그모이드 함수가 계단 함수가 다른 점이 무엇인지 알아보도록 하겠습니다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1920px-Logistic-curve.svg.png" alt="logistic" style="zoom: 25%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/Logistic_function">wikipedia - Logistic function</a></p>

<p align="center"><img src="https://mathworld.wolfram.com/images/interactive/TanhReal.gif" alt="hyper" style="zoom:110%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://mathworld.wolfram.com/HyperbolicTangent.html">mathworld.wolfram.com</a></p>

두 함수는 계단 함수의 단점이었던 불연속을 해결했습니다. 계단 함수와 시그모이드 함수의 공통점은 **비선형 함수(Non-linear)**라는 점입니다. 활성화 함수는 비선형 함수를 사용해야 합니다. 활성화 함수가 선형 함수이면 안되는 이유는 무엇일까요? 선형인 활성화 함수 $l(x) = ax + b$ 가 있다고 해보겠습니다. 이런 활성화 함수를 사용하여 3개의 층을 반복해서 쌓는다면 최종적인 활성화 함수는 $l(l(l(x))) = l^3(x) = a(a(ax+b)+b)+b = a^3x+a^2b+ab+b$가 됩니다. $a^3 = c, d = a^2b+ab+b$라고 하면 $l^3(x) = cx+d$로 여전히 같은 형태의 함수를 사용하는 것과 같기 때문에 층을 깊게 쌓는 의미가 없게 됩니다.

로지스틱 함수와 하이퍼탄젠트 함수의 수식은 다음과 같습니다.


$$
\begin{aligned}
\text{Logistic} &: \frac{1}{1+e^{-x}} \\
\text{Hypertangent} &: \frac{e^x-e^{-x}}{e^x+e^{-x}} = \frac{e^{2x}-1}{e^{2x}+1}
\end{aligned}
$$


### ReLU Function

시그모이드 함수 역시 한계점을 가지고 있습니다. 문제는 **기울기 소실(Gradient vanishing)**입니다. 기울기 소실은 시그모이드 함수를 활성화 함수로 사용하여 층을 깊게 쌓았을 때 학습이 잘 되지 않는 현상입니다. 이런 현상이 왜 발생하는지 알아보겠습니다.

우선 로지스틱 함수 $L(x)$를 미분한 함수의 수식에 대해서 알아보겠습니다.


$$
L^\prime(x) = \bigg(\frac{1}{1+e^{-x}}\bigg)^\prime = \frac{e^x}{(1+e^{-x})^2}
$$


위 함수의 그래프는 아래와 같이 생겼습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91560184-5d5bd900-e974-11ea-8c02-2a182c6a7c93.png" alt="logistic_deri" style="zoom:67%;" /></p>

위 함수는 최댓값이 0.25밖에 되지 않고 $x$가 5보다 크거나 -5보다 작을 때는 거의 0에 가깝습니다. 최댓값이 1보다 작으면 추후 오차 역전파(Backpropagation)를 통해 정보를 전달하는 과정에서 정보가 유실됩니다. 특히 층을 깊게 쌓는다면 정보가 거의 모두 유실되는 사태가 발생합니다.

하이퍼탄젠트 함수는 어떻게 될까요? 하이퍼탄젠트 함수 $\tanh$를 미분한 함수의 수식은 다음과 같습니다.


$$
\tanh^\prime(x) = \bigg(\frac{e^x-e^{-x}}{e^x+e^{-x}}\bigg)^\prime = \frac{4e^{2x}}{(1+e^{2x})^2}
$$


위 함수의 그래프는 아래와 같이 생겼습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91560164-52a14400-e974-11ea-8bf4-bbfc7fd42deb.png" alt="hypertangent_deri" style="zoom: 67%;" /></p>

함수의 최댓값이 높아져 $x=0$ 에서의 미분값이 1로 증가하였습니다. 덕분에 로지스틱 함수 보다는 정보를 잘 전달합니다 (이런 이유 때문에 RNN에서는 활성화 함수로 $\tanh$ 함수를 사용합니다). 하지만 여전히 $x$가 0에서 멀어질수록 미분값이 0에 가까워지기 때문에 정보를 제대로 전달할 수 없게 됩니다. 실제로 $0.5$만큼을 전달하는 3층으로 전달하더라도 $0.5^3 = 0.125$ 밖에 되지 않습니다. 이렇게 시그모이드 함수를 활성화 함수로 사용할 때 역전파시 학습이 제대로 진행되지 않는 현상을 **기울기 소실**이라고 합니다.

기울기 소실 문제를 극복하기 위해서 등장한 함수가 바로 **ReLU(Rectified Linear Unit)함수**입니다. ReLU함수는 입력값이 0보다 작을 경우에는 0을 반환하고, 0보다 클 경우에는 입력값을 그대로 반환합니다. 아래는 ReLU함수를 수식으로 나타낸 것입니다.
$$
h(x) = \begin{cases} 0 \qquad (x \leq 0) \\
x \qquad (x > 0) \end{cases}
$$

아래는 ReLU함수의 그래프를 나타낸 것입니다.

<p align="center"><img src="https://miro.medium.com/max/1225/0*g9ypL5M3k-f7EW85.png" alt="ReLU" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://medium.com/@sonish.sivarajkumar/relu-most-popular-activation-function-for-deep-neural-networks-10160af37dda">medium.com</a></p>

ReLU함수는 $x$가 0보다 클 때는 항상 미분값이 1이 되므로 층이 깊어지더라도 층 사이에 정보를 그대로 넘겨줄 수 있게 됩니다. 게다가 미분값이 항상 0과 1로 나와 복잡하지 않기 때문에 연산을 빠르게 할 수 있다는 점도 ReLU함수의 장점입니다. ReLU함수는 이런 장점 덕분에 은닉층에서 가장 많이 사용되는 활성화 함수가 되었습니다. 물론 ReLU함수에게도 0이하의 값이 그대로 보존되지 않고 버려지는 문제가 있습니다. 이를 보완하기 위한 Leaky ReLU함수로 고안되어 사용되고 있습니다. Leaky ReLU함수 $h_\text{Leaky}(x)$의 수식은 다음과 같습니다.


$$
h_\text{Leaky}(x) = \begin{cases} ax \qquad (x \leq 0) \\
x \qquad (x > 0) \end{cases}
$$


위 수식에서 일반적으로 $a=0.01$을 사용하며, 그래프는 다음과 같이 그릴 수 있습니다.

<p align="center"><img src="https://miro.medium.com/max/1225/1*siH_yCvYJ9rqWSUYeDBiRA.png" alt="leaky" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e">medium.com</a></p>

### Softmax

은닉층(Hidden Layer)의 활성화 함수로는 일반적으로 ReLU함수 혹은 Leaky ReLU와 같은 ReLU함수를 변형한 함수가 주로 사용됩니다. 하지만 출력층의 활성화 함수는 우리가 하고자 하는 작업에 맞게 조정해주어야 합니다. 일반적으로 회귀( Regression), 즉 연속형 변수에 대한 예측값을 출력하는 경우에는 활성화 함수로 항등함수 $h_\text{reg}(x) = x$ 를 사용합니다.

이진 분류의 경우에는 입력값을 받아 $0$ 혹은 $1$ 의 값을 출력하는 것이므로 주로 로지스틱 함수를 많이 사용합니다. 그렇다면 인스턴스를 다중 레이블로 분류하는 경우에는 어떤 활성화 함수를 사용하는 것이 좋을까요? 이런 질문에 대한 답으로 나온 것이 바로 소프트맥스(Softmax) 함수입니다. 소프트맥스 함수는 이진 분류에서 사용하는 로지스틱 함수를 다중 분류에서 사용할 수 있도록 일반화한 함수입니다. 소프트맥스의 함수는 다음과 같습니다.


$$
y_k = \frac{\exp(a_k)}{\sum^n_{i=1}\exp(a_i)}
$$


소프트맥스도 함수도 사용할 때 주의해야 할 점이 있습니다. 소프트맥스 함수가 지수함수이기 때문에 $a$ 값이 커지게 되면 $\exp(a)$ 값이 매우 커지게 됩니다. `__int32`가 최대로 나타낼 수 있는 숫자는 $2,147,483,647$ 입니다 $a = 22$ 만 되더라도 표현할 수 있는 값 이상이 되어 오버플로(Overflow)현상이 발생합니다. 또한 부동소수점 표기 특성상, 작은 숫자를 큰 값으로 나누면 수치가 불안정해지는 문제도 발생하게 됩니다.

이런 문제를 해결하기 위해서 실제로 소프트맥스 함수를 사용하기 위해서는 상수 $C$를 곱해주어 스케일을 조정해주는 과정이 필요합니다. 실제로 구현되어 있는 소프트맥스 함수의 수식은 아래와 같습니다.



$$
\begin{aligned}
y_k &= \frac{\exp(a_k)}{\sum^n_{i=1}\exp(a_i)} = \frac{C\exp(a_k)}{C\sum^n_{i=1}\exp(a_i)} \\
&= \frac{\exp(a_k +\log C)}{\sum^n_{i=1}\exp(a_i + \log C)} \\
&= \frac{\exp(a_k +C^\prime)}{\sum^n_{i=1}\exp(a_i + C^\prime)}
\end{aligned}
$$



위 식에서 $C^\prime = \log C$로, $C^\prime$에는 0보다 작은 값이면 어떤 값을 대입하든 상관 없지만 오버플로를 막기 위해서 일반적으로 $a_i \{i=1, \cdots ,n\}$ 중 가장 큰 값에 $-1$ 을 곱해준 값을 사용합니다. 예를 들어, $a_i = [1000, 1050, 1100]$이면 $C^\prime = -1100$ 이 됩니다.

소프트맥스 함수의 출력값은 항상 $[0,1]$ 범위 내에 있으며 모든 출력값을 더한 값이 1이 되는, 즉 $\sum^n_{i=1}y_i = 1$ 인 특징이 있습니다. 이런 성질 덕분에 소프트맥스의 출력값을 확률(Probability)로도 해석할 수 있으며 다중 레이블에 대한 확률이 필요한 경우에 소프트맥스 함수를 사용하기도 합니다.