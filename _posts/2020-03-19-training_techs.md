---
layout: post
title: 옵티마이저(Optimizer)
category: Deep Learning
tag: Deep-Learning
---





이번 게시물은 *["An overview of gradient descent optimization algorithms"](https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent)* 와 그의 번역인 [*"Gradient Descent Overview"*](https://brunch.co.kr/@chris-song/50)를 참조하여 작성하였습니다.

# Optimizer

가중치 초기화(Parameter initialization)를 통해서 시작점을 정했으니 이제는 어떤 방향으로 나아가야 할 지에 대해서 알아보겠습니다. 최적화하는 방법을 결정하는 것은 **옵티마이저(Optimizer)**라는 장치입니다. 이번 시간에는 바로 이 옵티마이저에 대해 알아보도록 하겠습니다.

일단 옵티마이저의 종류부터 알아보겠습니다. 옵티마이저의 종류는 다음과 같은 것들이 있습니다.



<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91630397-18838100-ea0c-11ea-8f90-515ef74599f1.png" alt="optimizer" style="zoom: 50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.slideshare.net/yongho/ss-79607172?from_action=save">slideshare.net/yongho</a></p>

위 그림에서 위쪽 방향은 어떤 방향으로 최적화를 진행할 것인지를 개선합니다. 아래쪽 방향은 갱신할 때마다 얼마나 진해 

우리가 알고 있는 경사 하강법(Gradient descent)은 위 그림에서 GD입니다. 확률적 경사 하강법부터 하나하나 보도록 하겠습니다.

## 확률적 경사 하강법(Stochastic Gradient Descent, SGD)

일반적인 경사 하강법은 파라미터를 한 번 갱신할 때마다 모든 데이터셋을 사용합니다. 하지만 데이터셋이 클 때는 이런 방법은 매우 많은 시간을 소요합니다. 예를 들어, 사용할 데이터셋이 100만건이라면 스텝을 밟아갈 때마다 100만건의 데이터를 전부 사용하게 되는 것이지요. 이런 문제를 해결하기 위해서 **확률적 경사 하강법(Stochastic Gradient Descent, SGD)**이라는 방법이 고안되었습니다.

### 확률적 경사 하강법

확률적 경사 하강법에서는 주어진 데이터셋에서 하나의 데이터를 선택하여 스텝을 나갈 때마다 이 데이터를 기준으로 경사 하강법으로 개선해나갑니다. 파라미터를 개선할 때마다 다른 데이터가 선택됩니다. 매번 하나의 인스턴스만을 선택하여 학습하기 때문에 일반적인 경사하강법 보다는 학습 속도가 훨씬 빠르다는 장점이 있습니다. 하지만 선택된 데이터가 경향성을 벗어나는 경우에는 오차가 급격히 증가하며 이 때문에 학습이 불안정하다는 단점을 가지고 있습니다. 아래는 경사 하강법과 확률적 경사 하강법을 표현한 그림입니다.

<p align="center"><img src="https://1.bp.blogspot.com/-KX6skkRmUvE/W9_DjB0XLSI/AAAAAAAAGoY/ccvrYtO_3HQZhU_U084XY_uAnfQ5QbumACLcBGAs/s1600/Screen%2BShot%2B2018-11-04%2Bat%2B6.16.45%2BPM.png" alt="sgd"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://ankit-ai.blogspot.com/2018/11/optimization-algorithms-for-machine.html">ankit-ai.blogspot.com</a></p>

확률적 경사 하강법의 수식은 아래와 같습니다.


$$
\theta \leftarrow \theta - \eta \nabla_\theta J(\theta;x^i;y^i)
$$


### 미니배치 경사 하강법

이런 확률적 경사 하강법의 단점을 보완하기 위해서 하나의 인스턴스 대신 $n$개의 인스턴스로 구성된 미니배치를 활용하는 **미니배치 경사 하강법(Minibatch gradient descent)**도 있습니다. 여러 개의 데이터를 사용하기 때문에 확률적 경사 하강법의 학습시 불안정성을 어느정도 줄일 수 있다는 단점이 있습니다.

$n$개의 인스턴스로 구성된 미니배치를 사용하는 미니배치 경사 하강법의 수식은 아래와 같이 나타낼 수 있습니다.


$$
\theta \leftarrow \theta - \eta \nabla_\theta J(\theta;x^{i:i+n};y^{i:i+n})
$$


확률적 경사 하강법의 단점은 비등방성[^1] 함수, 즉 방향에 따라 기울기 변화가 다른 함수에서는 비효율적인 탐색 경로를 보여준다는 단점을 가지고 있습니다. 확률적 경사 하강법과 경사 하강법을 비교했던 그림에서 확률적 경사 하강법은 지그재그를 그리며 최적화를 하고 있습니다, 그렇기 때문에 최소인 점에 이르기까지 더 많은 횟수의 파라미터 갱신이 필요하게 됩니다.

물론 하나의 데이터만을 사용하기 때문에 일반적인 경사 하강법 보다는 갱신하는 횟수가 많아지더라도 전체 시간은 훨씬 더 빠릅니다. 하지만 이렇게 지그재그로 탐색하는 것은 학습 시간 측면에서 여전히 비효율적이며, 국지적 최소점(Local minima)에 빠질 확률이 높다는 단점을 가지고 있습니다.

## 모멘텀(Momentum)

**모멘텀(Momentum)**은 확률적 경사 하강법이 가지고 있는 이런 비효율성을 극복하기 위해 고안된 방법입니다. 모멘텀은 물리에서 '운동량'을 뜻하는 단어입니다. 단어의 의미 그대로 최적화에 운동량을 적용합니다. 기울기가 심한 곳에서 공의 운동량이 커져서 더 빨라지는 것처럼 그래디언트의 변화가 심한 구간에서는 값을 더 많이 개선합니다. 반대로 기울기가 완만한 곳에서는 공이 느리게 굴러가는 것처럼 그래디언트의 변화가 더딘 구간에서는 더 세심하게 값을 개선합니다.

아래는 비등방성 함수에서 일반적인 확률적 경사 하강법과 모멘텀을 적용한 확률적 경사 하강법으로 최저점을 탐색하는 과정을 도식화한 이미지입니다.

<p align="center"><img src="https://www.researchgate.net/publication/333469047/figure/fig1/AS:764105438793728@1559188341202/The-compare-of-the-SGD-algorithms-with-and-without-momentum-Take-Task-1-as-example-The.png" alt="momentum" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.researchgate.net/figure/The-compare-of-the-SGD-algorithms-with-and-without-momentum-Take-Task-1-as-example-The_fig1_333469047">researchgate.net</a></p>

위 그림에서 모멘텀을 적용한 경사하강법은 그렇지 않았을 때보다 더 적은 횟수만으로 최저점에 다가간 것을 알 수 있습니다. 모멘텀이 더 빠르게 나아갈 수 있는 비결은 무엇일까요? 수식을 통해 모멘텀이 파라미터를 갱신하는 방법을 알아보겠습니다.


$$
\begin{aligned}
v_t &= \gamma v_{t-1} + \eta\nabla_\theta J(\theta) \\
\theta &\leftarrow \theta - v_t
\end{aligned}
$$


위 식에서 $v_t$는 각 지점에서의 속도입니다. 처음 $(v_0)$에는 0에서 시작하여 이전 그래디언트의 변화 $\eta\nabla_\theta J(\theta)$ 가 크거나 작게 되면 그 값이 다음에는 $v_{t-1}$이 되어 다음 변화에 영향을 끼치게 됩니다. $\gamma$는 모멘텀의 영향을 얼마나 줄 것인지에 대한 하이퍼파라미터로 이 값이 크면 이전 변화의 영향을 많이 받게 됩니다.



## NAG

모멘텀의 기능을 개선한 옵티마이저로 **NAG(Nesterov Accelerated Gradient)**가 있습니다. NAG의 식은 다음과 같습니다.


$$
\begin{aligned}
v_t &= \gamma v_{t-1} + \eta\nabla_\theta J(\theta - \gamma v_{t-1}) \\
\theta &\leftarrow \theta - v_t
\end{aligned}
$$


NAG 옵티마이저는 그래디언트의 변화를 그 자리에서 계산하지 않고 모멘텀에 의해서 이동한 곳에서 계산하는 방법입니다. 모멘텀에 의해서 변화된 만큼이 $\gamma v_{t-1}$이므로 이 값을 빼준 상태에서 그래디언트의 변화를 구하게 됩니다. 기존 모멘텀은 최소점 부근에서 손실 함수의 그래디언트 변화가 급격할 경우 최소점을 지나쳐 버리는 위험이 있었습니다. 하지만 NAG는 이런 위험도를 낮춤으로써 오히려 더욱 빠르게 최소점에 도달할 수 있다는 장점을 가지고 있습니다. 다양한 NAG에 대한 설명은 [이곳](https://jlmelville.github.io/mize/nesterov.html)을 참고하면 좋습니다.

## Adagrad

**Adagrad(아다그라드)**는 'Ada'ptive 'Grad'ient의 줄임말로, 말 그대로 각 파라미터에 '적합한 그래디언트'를 찾아 적용하는 옵티마이저입니다. Adagrad는 각 파라미터마다 다른 학습률(Learning rate)을 적용합니다. 자주 등장하는 특성의 파라미터에는 낮은 학습률을 적용하고, 가끔 등장하는 특성의 파라미터는 높은 학습률을 적용하여 빠르게 학습시킵니다. 이러한 방법은 희소한 데이터에 대하여 확률적 경사 하강법보다 특히 강건하다는(Robust) 장점을 가지고 있습니다.

Adagrad의 수식을 살펴보겠습니다. Adagrad는 각 특성의 파라미터마다 다른 값을 적용하므로 $i$ 번째 특성의 그래디언트를 $g_{t,i}$를 아래와 같이 표기해줍니다.


$$
g_{t,i} = \nabla_\theta J(\theta_{t,i})
$$


각 파라미터 $\theta_i$의 갱신은 아래와 같은 식을 통해서 진행됩니다. 아래 식에서 $G_t$는 $t$번째 학습에서 $\theta_i$에 대한 그래디언트 제곱의 합을 나타낸 것입니다. $\epsilon$은 $G_t$가 0일 경우 분모가 0이 되는 현상을 막기 위한 아주 작은 상수로 일반적으로는 $10^{-8}$을 사용합니다.


$$
\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t,ii}+\epsilon}}\cdot g_{t,i}
$$


이를 모든 파라미터에 대한 식으로 일반화하면 다음과 같습니다.


$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t
$$


위 식에서 $\odot$은 행렬의 원소별 곱셈을 의미합니다. 아래에서 일반적인 행렬곱과 원소별 곱을 비교해보겠습니다.



>$$
>\begin{aligned}
>&\left[\begin{array}{cc} 1 & 2 \\ 3 & 4\end{array}\right] \cdot \left[\begin{array}{cc} 1 & 2 \\ 3 & 4\end{array}\right] = \left[\begin{array}{cc} 1\times1+2\times3 & 1\times2+2\times4 \\ 3\times1+4\times3 & 3\times2+4\times4\end{array}\right] = \left[\begin{array}{cc} 7 & 10 \\ 15 & 22\end{array}\right] \\
>&\left[\begin{array}{cc} 1 & 2 \\ 3 & 4\end{array}\right] \odot \left[\begin{array}{cc} 1 & 2 \\ 3 & 4\end{array}\right] = \left[\begin{array}{cc} 1\times1 & 2\times2 \\ 3\times3 & 4\times4\end{array}\right]=\left[\begin{array}{cc} 1 & 4 \\ 9 & 16\end{array}\right]
>\end{aligned}
>$$



아다그라드의 가장 큰 장점은 따로 학습률을 조정해주지 않아도 알아서 적절한 학습률을 적용한다는 데에 있습니다. 하지만  단점도 역시 존재합니다. $G_t$의 성분에 계속 양수가 더해지기 때문에 학습을 계속할수록 $G_t$값이 커지게 되고 학습률은 점점 줄어들게 됩니다. 이 때문에 학습을 아주 많이 한다면 파라미터가 거의 갱신되지 않는 현상이 발생합니다.



## Adadelta & RMSprop

### Adadelta

이런 문제를 해결하기 위해서 등장한 것이 **Adadelta**와 **RMSprop**입니다. 먼저 Adadelta부터 알아보겠습니다. Adagrad는 $G_t$가 너무 커져버려서 학습률이 거의 0에 가까워지는 단점이 있었습니다. Adadelta는 이런 현상을 막기 위해서 모든 그래디언트를 제곱합하는 $G_t$대신 이전 $w$개 만큼을 제곱하여 더한 값의 평균인 $E[g^2]_t$를 사용합니다. Adadelta에서 이를 구하는 수식은 다음과 같습니다. 아래 식에서 $\gamma$은 비율을 조정하는 하이퍼파라미터이며 일반적으로는 $0.9$ 정도로 설정합니다.


$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g^2_t
$$


Adagrad의 수식을 먼저 써보고 Adadelta가 어떤 방식으로 이를 개선하는지 알아보겠습니다. 위에서 살펴본 Adagrad의 수식은 아래와 같았습니다.


$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t
$$


Adadelta는 더 이상 $G_t$를 사용하지 않으므로 아래와 같이 식이 바뀌게 됩니다.


$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}\cdot g_t
$$


아래 항은 파라미터의 제곱합에 루트를 취해준 것으로 간단히 제곱합의 제곱근(Root Mean Square)을 뜻하는 $RMS[g]_t$로 표현해 줄 수 있습니다. 이를 사용하여 위 식을 다음과 같이 다시 쓸 수 있습니다.

Adagrad의 저자들은 위와 같이 (확률적 경사 하강법, 모멘텀, Adagrad에서도 사용했던) 그래디언트를 사용하여 식을 서술하는 방식은 단위가 다르다고 말했습니다. 그러면서 위에서 사용했던 식을 다음과 같이 바꿔써야한다고 제안하였습니다. 아래에 있는 식에서는 그래디언트 대신 파라미터의 변화량으로 표기합니다. 의미상으로는 거의 차이가 없지만 단위에 차이가 있습니다.


$$
E[\Delta\theta^2]_t = \gamma E[\Delta\theta^2]_{t-1} + (1-\gamma)\Delta\theta^2_t \\
RMS[\Delta\theta]_t = \sqrt{E[\Delta\theta^2]_t+\epsilon} \\
\theta_{t+1} = \theta_t - \frac{RMS[\Delta\theta]_{t-1}}{RMS[g]_t}\cdot g_t
$$


### RMSprop

RMSprop은 Adadelta와 비슷한 시기에 등장한 옵티마이저입니다. 둘은 독립적으로 등장했음에도 Adagrad에서 학습률이 사라지는 문제를 해결하고자 고안된 것이기 때문에 형태가 거의 동일합니다. RMSprop 옵티마이저의 수식 형태는 다음과 같습니다.


$$
E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1g^2_t \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}\cdot g_t
$$


위 식을 보면 Adadelta의 첫번째 식에서 $\gamma = 0.9$로 고정된 것임을 알 수 있습니다. 이를 사용하여 그래디언트의 변화를 적용하는 것도 동일합니다. RMSprop을 개발한 힌튼 교수는 적정한 학습률 값으로 $\eta = 0.001$을 제시하였습니다.



## Adam

**Adam(Adaptive moment estimation, 아담)**은 지금까지 알아본 그래디언트 조정법(모멘텀, NAG 등)과 학습률 조정법(Adagrad, Adadelta, RMSprop)을 융합한 방법입니다. Adam은 그래디언트의 변화 $g_t$에 따라 중심 모멘텀(1차 모멘텀) $m_t$와 분산 모멘텀(2차 모멘텀) $v_t$를 아래와 같이 나타냅니다.


$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2_t
$$


Adam의 저자들은 이렇게 정의한 $m_t, v_t$가 초반 단계 $t \sim 0$ 에서 $\beta_1,\beta_2 \approx 1$ 일 때 0으로 편향되는 특성이 있음을 발견했습니다. 그리하여 실제 식에는 편향을 보정한 $\hat{m_t}, \hat{v_t}$를 정의하여 사용하였습니다.


$$
\hat{m_t} = \frac{m_t}{1-\beta^t_1}\\
\hat{v_t} = \frac{v_t}{1-\beta^t_2}
$$


이를 활용한 Adam의 수식은 아래와 같습니다.


$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\cdot \hat{m_t}
$$


저자들은 $\beta_1$의 적정한 기본값으로는 $0.9$를, $\beta_2$의 적정한 기본값으로는 $0.999$를, $\epsilon$의 적정한 값은 $10^{-8}$을 제안했습니다.

## And

아래는 다양한 모양의 손실 함수에서 지금까지 알아본 옵티마이저가 최적화하는 방향과 속도를 나타낸 것입니다.

<p align="center"><img src="https://rnrahman.com/img/visualising-optim/saddle.gif" alt="comparison1"  /></p>

<p align="center"><img src="https://rnrahman.com/img/visualising-optim/beale.gif" alt="comparison2"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://rnrahman.com/blog/visualising-stochastic-optimisers/">rnrahman.com/blog</a></p>

Adam외에도 AdaMax, Nadam등 다양한 옵티마이저가 있습니다. 이런 수많은 옵티마이저 중 어떤 것을 사용하는 것이 좋을까요? Adam은 이전의 모든 옵티마이저들이 가진 문제를 개선한 것이기에 SGD보다 대부분의 상황에서 좋은 성능을 보이기는 합니다. 하지만 *"공짜 점심은 없다"*라는 말은 모델뿐만 아니라 옵티마이저에도 적용되는 말입니다. 데이터셋마다 가장 좋은 성능을 보이는 옵티마이저가 다르고, 특히 하이퍼파라미터를 어떻게 설정하느냐에 따라서도 많은 변화가 있을 수 있습니다. 그렇기 때문에 최대한 많은 방법을 시도해보고 그 중 가장 좋은 성능을 보이는 것을 채택하는 것이 올바른 방법이라 할 수 있겠습니다.

[^1]: **비등방성**(anisotropy)은 방향에 따라 물체의 물리적 성질이 다른 것을 말한다. (출처 : 위키피디아 - 비등방성)