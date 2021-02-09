---
layout: post
title: 옵티마이저(Optimizer)
category: Deep Learning
tag: Deep-Learning
---



이번 게시물은 *["An overview of gradient descent optimization algorithms"](https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent)* 와 그의 번역인 [*"Gradient Descent Overview"*](https://brunch.co.kr/@chris-song/50)를 참조하여 작성하였습니다.

# Optimizer

[가중치 초기화(Parameter initialization)](https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/)를 통해서 시작점을 정했으니 하강하는 방법에 대해서 알아보겠습니다. **옵티마이저(Optimizer)**는 최적화 방법을 결정해주는 방식입니다. 이번 게시물에서는 옵티마이저에 대해 알아보도록 하겠습니다. 우선 아래 이미지를 통해 어떤 옵티마이저가 있는지부터 알아보겠습니다. 

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91630397-18838100-ea0c-11ea-8f90-515ef74599f1.png" alt="optimizer" style="zoom: 50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.slideshare.net/yongho/ss-79607172?from_action=save">slideshare.net/yongho</a></p>

위 그림에서 위쪽 방향(파란색 화살표)은 어떤 방향으로 내려갈지를 개선하는 과정에서 탄생한 옵티마이저입니다. 아래쪽 방향(빨간색 화살표)는 얼마나 내려갈지(학습률)를 개선하는 과정에서 탄생한 옵티마이저이지요. 우리에게 익숙한 **경사 하강법(Gradient descent, GD)**을 개선한 확률적 경사 하강법부터 알아보도록 하겠습니다.



## 확률적 경사 하강법(Stochastic Gradient Descent, SGD)

경사 하강법은 학습에 모든 데이터를 다 사용합니다. 이 방법은 데이터셋이 커질수록 훨씬 더 많은 시간을 필요로합니다. 사용할 데이터셋이 100만건이라면 100만건의 데이터에 대해 모두 순전파를 계산하고 손실을 구한 뒤 이를 개선하는 방향으로 파라미터가 한 번 갱신됩니다. 당연히 시간이 많이 걸릴 수밖에 없겠지요. 이런 문제를 해결하기 위해서 등장한 것이 **확률적 경사 하강법(Stochastic Gradient Descent, SGD)**입니다.

### 확률적 경사 하강법

확률적 경사 하강법에서는 전체 데이터셋에서 하나의 데이터를 선택합니다. 그리고 이 데이터만을 사용하여 경사 하강법으로 개선해나갑니다. 대신 파라미터를 개선할 때마다 다른 데이터가 선택됩니다. 하나의 인스턴스만을 선택하여 학습하기 때문에 보통의 경사 하강법 보다 학습 속도가 훨씬 빠릅니다. 하지만 선택된 데이터가 아웃라이어일 수도 있겠지요. 이런 경우 오차가 급격히 증가하며 학습이 불안정해지는 단점을 가지고 있습니다. 아래는 확률적 경사 하강법과 경사 하강법을 비교하여 나타낸 그림입니다.

<p align="center"><img src="https://1.bp.blogspot.com/-KX6skkRmUvE/W9_DjB0XLSI/AAAAAAAAGoY/ccvrYtO_3HQZhU_U084XY_uAnfQ5QbumACLcBGAs/s1600/Screen%2BShot%2B2018-11-04%2Bat%2B6.16.45%2BPM.png" alt="sgd"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://ankit-ai.blogspot.com/2018/11/optimization-algorithms-for-machine.html">ankit-ai.blogspot.com</a></p>

오른쪽에 있는 보통의 경사 하강법은 5번의 파라미터 갱신을 통해 최적점을 찾았습니다. 하지만 학습 데이터셋이 컸다면 한 번 갱신하는데 엄청난 시간이 걸렸을 것입니다. 반대로 확률적 경사 하강법은 17번이나 파라미터 갱신한 끝에 최적점을 찾았네요. 하지만 17개의 인스턴스만을 계산하여 얻어낸 결과입니다. 한 번 갱신할 때마다 데이터를 하나만 사용하기 때문에 계산이 매우 빠르지요. 확률적 경사 하강법의 수식은 아래와 같습니다.

$$
\theta \leftarrow \theta - \eta \nabla_\theta J(\theta;x^i;y^i)
$$


### 미니배치 경사 하강법

데이터를 하나만 사용하는 것은 상당히 불안합니다. 아웃라이어가 학습에 사용되는 날에는 최적점으로부터 엄청나게 멀어지게 되지요. 이런 단점을 해결하기 위한 것이 **미니배치 경사 하강법(Mini-batch gradient descent)**입니다. 하나의 인스턴스는 너무 불안정하니 $n$ 개의 인스턴스로 이루어진 미니배치를 학습 단위로 사용하자는 것이 미니배치 경사 하강법의 기본 아이디어입니다. 여러 개의 데이터를 사용하기 때문에 불안정성을 줄일 수 있다는 장점이 있습니다. 만약 배치 사이즈가 $100$이라면 그 안에 있는 데이터가 모두 아웃라이어일 확률은 매우 적어지겠지요.

$n$개의 인스턴스로 구성된 미니배치를 사용하는 미니배치 경사 하강법의 수식은 아래와 같이 나타낼 수 있습니다.


$$
\theta \leftarrow \theta - \eta \nabla_\theta J(\theta;x^{i:i+n};y^{i:i+n})
$$



### 한계점

확률적 경사 하강법은 한계점도 가지고 있습니다. 비등방성[^1] 함수, 즉 방향에 따라 기울기 변화가 다른 함수에서 비효율적인 탐색 경로를 보여준다는 것이지요. 위 그림에서도 확률적 경사 하강법은 최적점까지 가는 데 훨씬 더 많은 갱신 횟수를 필요로 했습니다. 일반적인 경사 하강법보다는 빠르지만 **여전히 비효율적**이지요. 게다가 경향성에서 많이 벗어나는 데이터가 선택될 경우 국지적 최소점(Local minima)에 빠질 수 있다는 단점도 가지고 있습니다.



## 모멘텀(Momentum)

이런 한계점을 극복하기 위해 등장한 방법이 **모멘텀(Momentum)**입니다. 모멘텀의 우리말 뜻은 '운동량'입니다. 말 그대로 운동량을 최적화 과정에 적용합니다. 기울기가 큰 곳, 즉 그래디언트의 변화가 심한 구간에서는 값을 더 많이 개선합니다. 반대로 완만한 곳에서는 파라미터를 미세하게 개선해나가지요.

비등방성 함수에서 모멘텀을 적용했을 때와 그렇지 않을 때 확률적 경사 하강법이 어떻게 파라미터를 갱신하는지 알아보겠습니다.

<p align="center"><img src="https://www.researchgate.net/publication/333469047/figure/fig1/AS:764105438793728@1559188341202/The-compare-of-the-SGD-algorithms-with-and-without-momentum-Take-Task-1-as-example-The.png" alt="momentum" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.researchgate.net/figure/The-compare-of-the-SGD-algorithms-with-and-without-momentum-Take-Task-1-as-example-The_fig1_333469047">researchgate.net</a></p>

모멘텀을 적용했을 때 기울기가 급격하게 변하는 쪽(X축 방향)으로 더 많이 움직이는 것을 볼 수 있습니다. 덕분에 더 적은 갱신 횟수로도 최적점 가까이에 다가갈 수 있었지요. 수식을 통해 어떻게 최적점으로 빠르게 다가갈 수 있는지 알아보겠습니다.



$$
\begin{aligned}
v_t &= \gamma v_{t-1} + \eta\nabla_\theta J(\theta) \\
\theta & \leftarrow \theta - v_t
\end{aligned}
$$



위 식에서 $v_t$는 각 지점에서의 속도입니다. 확률적 경사 하강법의 식에 $\gamma \cdot v_{t-1}$ 이 더해졌습니다. 이전 갱신에서 구한 속도가 다음 속도에 영향을 미치게 되지요. $\gamma$ 는 모멘텀의 영향력을 얼마나 크게 할 것인지에 대한 하이퍼파라미터입니다. $\gamma = 0$ 이면 모멘텀을 적용하지 않은 확률적 경사 하강법과 같아지게 되며 이 값이 커질수록 모멘텀의 영향을 더 많이 받게 됩니다.



## NAG

모멘텀을 다르게 적용하는 **NAG(Nesterov Accelerated Gradient)**라는 옵티마이저도 있습니다. NAG의 식은 다음과 같습니다.


$$
\begin{aligned}
v_t &= \gamma v_{t-1} + \eta\nabla_\theta J(\theta - \gamma v_{t-1}) \\
\theta &\leftarrow \theta - v_t
\end{aligned}
$$

NAG 옵티마이저는 그래디언트의 변화를 그 자리에서 계산하지 않고 모멘텀에 의해서 이동한 곳에서 계산합니다. 모멘텀에 의해서 변화된 $\gamma \cdot v_{t-1}$ 을 빼준 곳, 즉 $(\theta - \gamma v_{t-1})$ 에서 그래디언트의 변화를 구합니다. NAG를 고안한 이유는 기존 모멘텀 적용 방식에 한 가지 문제가 있기 때문입니다. 최소점 부근에서 그래디언트 변화가 갑자기 클 때 최소점을 지나쳐 버리게 되지요. 하지만 NAG는 이전 위치에서 그래디언트를 계산하여 이런 사태에 대한 위험도를 낮추었습니다. 이런 개선을 통해 NAG는 오히려 최소점에 더욱 빠르게 도달할 수 있다는 장점을 가지고 있습니다. 다양한 NAG에 대한 설명은 [이곳](https://jlmelville.github.io/mize/nesterov.html)을 참고하면 좋습니다.

지금까지 확률적 경사 하강법과 이를 개선한 모멘텀, NAG를 알아보았습니다. 이 옵티마이저들은 모두 학습 방향을 어떻게 조정할 지 개선하는 과정에서 나왔습니다. 텐서플로우와 파이토치에서는 위 옵티마이저를 사용하기 위해서 아래와 같은 메서드를 제공하고 있습니다.

```python
#TensorFlow
tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD', **kwargs
)

#Pytorch
torch.optim.SGD(
    params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False
)
```



## Adagrad

다음으로 **학습률(Learning rate)**을 조정하여 성능을 개선한 몇 가지 옵티마이저에 대해서 알아보겠습니다. 이렇게 개선된 옵티마이저의 대표격이 바로 **Adagrad(아다그라드)**입니다. Adagrad는 'Ada'ptive 'Grad'ient의 줄임말입니다. 적응형 그래디언트, 즉 파라미터마다 다른 학습률을 적용하겠다는 뜻이지요. 

Adagrad는 자주 등장하는 특성의 파라미터에 대해서는 낮은 학습률을, 가끔 등장하는 특성의 파라미터에 대해서는 높은 학습률을 적용합니다. 희소한 데이터에 대해 이 방법을 적용할 경우 훨씬 더 강건(Robust)하다는 장점이 있지요.

Adagrad의 수식을 살펴보겠습니다. 특성의 파라미터마다 다른 값을 적용하므로 $i$ 번째 특성의 그래디언트를 $g_{t,i}$를 아래와 같이 표기하겠습니다.


$$
g_{t,i} = \nabla_\theta J(\theta_{t,i})
$$



각 파라미터 $\theta_i$의 갱신은 다음과 같은 식을 통해서 진행됩니다.

$$
\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t,ii}+\epsilon}}\cdot g_{t,i}
$$



아래 식에서 $G_t$는 $t$번째 학습에서 $\theta_i$에 대한 그래디언트 제곱의 합을 나타낸 것입니다. $\epsilon$은 계산의 안정성을 위한 작은 상수로 일반적으로는 $10^{-8}$을 사용합니다. 이를 모든 파라미터에 대한 식으로 일반화하면 다음과 같습니다.



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



Adagrad는 학습률을 조정해주지 않아도 알아서 적절한 학습률을 적용한다는 장점이 있습니다. 하지만 단점도 존재합니다. $G_t$ 에 계속 양수가 더해지기 때문에 학습이 진행될수록 $G_t$값이 커지고 학습률이 점점 줄어들게 되지요. 이 때문에 많은 반복 학습 이후에는 파라미터가 거의 갱신되지 않는 문제가 발생합니다.



## Adadelta & RMSprop

이런 문제를 해결하기 위해서 등장한 옵티마이저가 Adadelta와 RMSprop입니다.

### Adadelta

먼저 **Adadelta**부터 알아보겠습니다. Adadelta는 학습률이 0에 가까워지는 문제를 해결하기 위해 모든 그래디언트의 제곱합인 $G_t$대신 이전 $w$개 만큼을 제곱하여 더한 값의 평균인 $E[g^2]_t$를 사용합니다. $E[g^2]_t$ 를 구하는 수식은 다음과 같습니다. $\gamma$은 비율을 조정하는 하이퍼파라미터이며 $0.9$ 정도로 설정합니다.


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

분모 부분은 그래디언트의 제곱합에 루트를 취해준 것으로 간단히 제곱합의 제곱근(Root Mean Square)을 뜻하는 $RMS[g]_t$ 를 사용하여 아래와 같이 표현할 수도 있습니다.


$$
\theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t}\cdot g_t
$$


Adagrad 논문의 저자는 단위를 맞추기 위해 그래디언트 $g_t$ 대신 파라미터의 변화량 $\Delta\theta$ 으로 표기하도록 하였습니다. 바꿔쓴 식은 아래와 같습니다.



$$
E[\Delta\theta^2]_t = \gamma E[\Delta\theta^2]_{t-1} + (1-\gamma)\Delta\theta^2_t \\
RMS[\Delta\theta]_t = \sqrt{E[\Delta\theta^2]_t+\epsilon} \\
\theta_{t+1} = \theta_t - \frac{\eta}{RMS[\Delta\theta]_t}\cdot g_t
$$



### RMSprop

**RMSprop**은 Adadelta와 비슷한 시기에 등장한 옵티마이저입니다. 독립적으로 등장했지만 Adagrad에서 학습률이 사라지는 문제를 해결하고자 고안된 것이기 때문에 형태가 거의 동일합니다. RMSprop 옵티마이저의 수식 형태는 다음과 같습니다.

$$
E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1g^2_t \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}\cdot g_t
$$


위 식이 Adadelta의 첫번째 식에서 $\gamma = 0.9$로 고정한 것과 같음을 알 수 있습니다. 그래디언트의 변화를 조정하는 것도 동일합니다. RMSprop을 제시한 제프리 힌튼은 적정한 학습률 값으로 $\eta = 0.001$을 제시하였습니다.



## Adam

**Adam(Adaptive moment estimation, 아담)**은 그래디언트 조정법(모멘텀, NAG 등)과 학습률 조정법(Adagrad, Adadelta, RMSprop)의 장점을 융합한 옵티마이저입니다. Adam은 그래디언트 $g_t$ 에 의해 변하는 중심(1차) 모멘텀 $m_t$와 분산(2차) 모멘텀 $v_t$를 사용합니다. 각 모멘텀은 다음과 같이 구할 수 있습니다.



$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2_t
$$



Adam의 저자들은 $m_t, v_t$ 를 단순히 위와 같이 설정할 때, 학습 초반 $(t \sim 0)$ 에서 $\beta_1,\beta_2 \approx 1$ 일 때의 값이 $0$ 으로 편향됨을 발견했습니다. 실제 식에는 편향을 보정한 $\hat{m_t}, \hat{v_t}$를 정의하여 사용하였습니다.

$$
\hat{m_t} = \frac{m_t}{1-\beta^t_1}\\
\hat{v_t} = \frac{v_t}{1-\beta^t_2}
$$


이를 활용한 Adam의 수식은 아래와 같습니다.


$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\cdot \hat{m_t}
$$


저자들은 $\beta_1$ 의 기본값으로는 $0.9$ 를, $\beta_2$의 기본값으로 $0.999$ 를, $\epsilon$의 적정한 값은 $10^{-8}$ 을 제안했습니다.

## Comparison

지금까지 알아본 옵티마이저가 아래는 2가지 형태의 손실 함수에서 최적화하는 방향과 속도를 시각화한 이미지입니다.

<p align="center"><img src="https://rnrahman.com/img/visualising-optim/saddle.gif" alt="comparison1"  /></p>

<p align="center"><img src="https://rnrahman.com/img/visualising-optim/beale.gif" alt="comparison2"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://rnrahman.com/blog/visualising-stochastic-optimisers/">rnrahman.com/blog</a></p>

지금까지 알아본 것 이외에도 AdaMax, Nadam, AdamW등 다양한 옵티마이저가 있습니다. 여러가지 옵티마이저 중 무엇을 사용하는 것이 좋을까요? Adam은 이전 옵티마이저의 문제를 해결하고자 나온 것이기에 초기에 고안된 SGD보다 대부분의 상황에서 좋은 성능을 보입니다.

하지만 *"공짜 점심은 없다"*라는 말은 모델뿐만 아니라 옵티마이저에도 적용되는 말입니다. 우선 데이터셋마다 가장 좋은 성능을 보이는 옵티마이저가 다릅니다. 손실 함수가 정말 단순한 형태를 가지고 있다면 Adam을 사용하는 것은 오히려 수렴을 늦추는 결과를 가져올 수도 있습니다. 같은 옵티마이저를 사용하더라도 하이퍼파라미터를 어떻게 설정하느냐에 따라서도 성능이 달라집니다. 그렇기 때문에 최대한 많은 방법을 시도해보고 그 중 가장 좋은 결과를 내주는 옵티마이저를 선택하는 것이 올바른 방법이라 할 수 있겠습니다.

[^1]: **비등방성**(anisotropy)은 방향에 따라 물체의 물리적 성질이 다른 것을 말한다. (출처 : 위키피디아 - 비등방성)