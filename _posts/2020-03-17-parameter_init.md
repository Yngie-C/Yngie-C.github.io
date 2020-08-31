---
layout: post
title: 가중치 초기화(Weight Initialization)
category: Deep Learning
tag: Deep-Learning
---





이번 게시물은 ["밑바닥부터 시작하는 딥러닝"](http://www.yes24.com/Product/Goods/34970929)과 [*"Weight Initialization Techniques in Neural Networks"*](https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78)를 참조하여 작성하였습니다.

# Parameter Initialization

이번에는 신경망의 **가중치 초기화(Weight initialization)**에 대해서 알아보겠습니다. 신경망 모델의 목적은 실제 레이블과 예측 레이블 값과 손실 함수(Loss function)로부터 도출되는 손실(Loss)을 최소화하는 과정, 즉 **파라미터를 최적화하는 과정(Parameter optimization)**입니다.

특성 변화에 따른 손실 함수의 그래프가 아래와 같이 생겼다고 해보겠습니다. 모델은 경사 하강법을 통해서 손실이 최저가 되는 점을 찾아가게 됩니다.하지만 아래 그림에서 함수가 어디에서 시작하느냐에 따라서 학습을 통해 도달하게 되는 최저점이 달라지게 됩니다.



<p align="center"><img src="https://miro.medium.com/max/1225/1*t4aYsxpCqz2eymJ4zkUS9Q.png" alt="init"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://medium.com/coinmonks/loss-optimization-in-scientific-python-d1efbbe87171">medium.com/coinmonks</a></p>

만약에 오른쪽 봉우리 어딘가에서 시작한다면 같은 지점에서 시작한다면 아래 새로 생긴 경로를 따라서 학습될 수도 있을 것입니다. 

![initial](https://user-images.githubusercontent.com/45377884/91631652-2fc76c00-ea16-11ea-98f2-3dc11c0f5f43.png)

위에서 $\theta_0, \theta_1$로 나타나는 가중치를 적절하게 초기화 해주는 것은 높은 성능의 신경망 모델 구축에 중요한 영향을 미치는 요소입니다. 위 그림에서와 같이 제대로 된 가중치 초깃값을 사용하지 않으면 경사 하강법을 사용하여 최저점을 찾아내기가 어렵게 됩니다. 그렇다면 가중치를 어떻게 초기화해주는 것이 좋을까요?



## Zero initialization

가장 단순하게 생각해볼 수 있는 것은 모든 신경망의 파라미터의 초깃값을 0으로 놓고 시작하는 것입니다. 하지만 파라미터의 값을 이렇게 설정하면 **안됩니다**. 더 자세히 말하면 모든 신경망의 파라미터가 같아서는 안됩니다.

신경망의 모든 파라미터의 값이 같다면 오차역전파(Backpropagation)로 갱신하는 파라미터의 값이 모두 같아지기 때문입니다. 신경망 노드들의 파라미터가 같다면 여러 개의 노드로 신경망을 구성하는 의미가 없어지게 됩니다. 이런 이유 때문에 초깃값은 모두 다른 값으로 설정해야합니다.

## Random Initialization

가중치마다 다른 값을 부여할 수 있는 방법 중 가장 쉽게 생각해볼 수 있는 방법은 정규분포 값을 각 가중치에 배정하는 것입니다. 먼저 표준편차가 $1$인 정규분포로 가중치를 초기화하고 활성화 함수로는 시그모이드(로지스틱) 함수를 사용한 5층짜리 신경망을 생각해 보겠습니다. 이런 신경망에서 가중치 분포를 그리면 아래와 같은 그래프가 나오게 됩니다. 

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91642462-852b6980-ea66-11ea-84cd-2cdbc3007db1.png" alt="gaussian" style="zoom:110%;" /></p>

가중치가 매우 작거나 큰 경우에는 시그모이드 함수로 도출하는 활성값이 0과 1에 가까운 값만 나오게 됩니다. 이런 값에서의 미분값은 거의 0이므로 학습이 일어나지 않는 [기울기 소실(Gradient vanishing)](https://yngie-c.github.io/deep learning/2020/03/10/neural_net_training/) 현상이 발생하기 때문입니다. 그렇다면 이번에는 너무 큰 값이나 작은 값이 나오지 않도록 표준편차를 $0.01$인 정규분포로 초기화한 후에 활성값 그래프를 그려보도록 하겠습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91642484-aab87300-ea66-11ea-94ba-f0236b3fe508.png" alt="gaussian2" style="zoom:110%;" /></p>

위 그림에서는 기울기 소실 효과는 발생하지 않았지만 대부분의 값이 0.5에 머무르고 있는 것을 알 수 있습니다. 위 **Zero initialization**에서도 말했던 것처럼 모든 함수의 활성화 값이 비슷하면 여러 노드를 가진 은닉층을 구성하거나, 은닉층을 여러 개로 구성하는 의미가 사라집니다.

## Xavier Initialization

**사비에르 초기화(Xavier initialization)**는 이런 문제를 해결하기 위해 고안된 초기화 방법입니다. 사비에르 초기화에서는 고정된 표준편차를 사용하지 않고 이전 은닉층의 노드 수에 맞추어 변화시킵니다. 이전 은닉층이 $n$개의 노드로 구성된 경우 사비에르 초기화는 $\frac{1}{\sqrt{n}}$ 을 표준편차로 하는 정규분포 값을 초깃값으로 배정합니다.

이렇게 배정한 초깃값으로 시작한 이전과 동일한 조건의 신경망에서의 활성화 값 그래프는 아래와 같습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91678679-24dd1a80-eb81-11ea-9637-abcdbcd3788c.png" alt="xavier" style="zoom:110%;" /></p>

위 그래프를 보면 층을 지나면서 활성값 분포 모양이 변하기는 합니다. 하지만 이전 두 방법보다는 훨씬 더 고르게 퍼져있는 것을 알 수 있습니다. 실제로 사비에르 초기화를 제시한 논문에서는 이전 은닉층의 개수뿐만 아니라 현재 은닉층의 개수 $m$도 고려하여 표준편차를 설정합니다. 이 때 사비에르 초기화가 사용하는 표준편차 수식은 $\frac{2}{\sqrt{n+m}}$입니다.



## He Initialization 

He 초기화(He Initialization)은 ReLU함수를 활성화 함수로 사용할 때 추천되는 초깃값입니다. 컴퓨터 비전(Computer vision) 분야의 대표적인 Pre-trained 모델인 VGG도 활성화 함수로 ReLU를 사용하고 있기 때문에 He 초깃화를 적용한 초깃값을 사용하고 있습니다. He 초기화는 $\frac{2}{\sqrt{n}}$를 표준편차로 하는 정규분포 값을 초깃값으로 배정합니다.

먼저 그래프를 보며 사비에르 초기화와 어떤 차이점이 있는 지를 알아보겠습니다. 아래는 활성화 함수가 ReLU함수로 구성된 5층의 신경망에서의 활성값이 어떻게 변하는 지를 나타낸 그래프입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91679856-5efbeb80-eb84-11ea-870b-9fe512141ca5.png" alt="xavier_relu" style="zoom:110%;" /></p>

첫 번째 층에서는 활성화 값이 골고루 분포되어 있지만 층이 깊어질수록 한쪽으로 분포가 치우치는 것을 볼 수 있습니다. 층이 5층보다 더 깊어진다면 0쪽으로 치우치는 값이 많아지게 되고, 이 때문에 기울기 소실이 다시 발생하게 됩니다. 그렇다면  He 초기화는 어떤 그래프를 그리는지 알아보겠습니다. 아래는 동일한 신경망에 He 초기화를 적용하여 나온 활성값을 나타낸 그래프입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/91680106-34f6f900-eb85-11ea-8abe-7bd830483f21.png" alt="he_relu" style="zoom:110%;" /></p>

He 초기화를 사용했더니 층이 깊어지더라도 초깃값이 거의 변화없이 고르게 분포되어 있는 것을 알 수 있습니다.



## Selection

이런 이유 때문에 사용하는 활성화 함수에 따라 적용해야 하는 초기화 방법이 달라집니다. 일반적으로 로지스틱 함수나 하이퍼탄젠트 $(\tanh)$ 등의 시그모이드 함수를 활성화 함수로 하는 경우에는 사비에르 초기화를 사용합니다. ReLU나 ReLU를 변형한 함수를 활성화 함수로 사용하는 경우에는 He 초기화를 사용합니다.