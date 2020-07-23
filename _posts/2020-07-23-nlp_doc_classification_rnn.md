---
layout: post
title: RNN을 활용한 문서 분류
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# RNN for Sentence Classification

순환 신경망(Recurrent Neural Network, RNN)을 사용하여 문서를 분류하는 방법도 있다. RNN에서는 토큰의 순서가 중요해진다. 각 자리에 있는 단어를 임베딩 벡터로 나타내 준 후에 각 단계마다의 은닉 상태(Hidden state)에 넣어주어 다음 단계로 보내준다. 마지막 토큰(End Token)이 들어왔을 때의 Hidden state는 모든 토큰의 정보를 가지고 있다. 이를 완전 연결 층(Fully Connected Layer)에 넘긴 후 소프트맥스를 사용하여 범주를 예측한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88295143-fe34f400-cd37-11ea-8ef5-4a5c941718eb.png" alt="rnn1" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

가장 단순한 RNN(Vanilla RNN)은 구조 특성상 문서를 분류할 때 뒷쪽에 위치하는 단어의 영향을 더 많이 받는다. 이런 현상은 기울기 소실(Gradient Vanishing)으로 인한 장기 의존성(Long-term Dependency) 등의 문제를 야기하는 원인이 된다. 이를 해결하기 위하여 과거 정보를 선택적으로 보존하기 위한 모델들이 고안되었다. RNN을 변형한 장단기 기억망(Long-Short Term Memory, LSTM)과 GRU(Gated Recurrent Unit) 등은 Hidden Node의 구조를 변형하여 기존에 발생하던 문제를 일부 해결할 수 있었다.

Hidden Node의 구조를 변형하는 방법 외에도 Hidden Layer를 변형하는 방법이 제시되었다. 첫 번째로 양방향(Bi-directional)으로 학습을 진행하여 한 쪽에서 정보가 소실되는 것을 보완하는 방법이 있다. 두 번째로는 아예 Hidden Layer를 여러 층(Multi-Layer)으로 쌓은 뒤 학습을 진행하는 방법이 제안되었다.



## Various RNN Architecture

위에서 제시된 방법 외에도 문서 분류를 위해서 고안된 다양한 방법들이 있다. 

첫 번째는 여러 데이터셋을 하나의 Hidden Layer에 처리하도록 하는 방법이다. 같은 Task를 수행하기 위한 여러가지 데이터셋을 학습시켜 모델의 성능을 향상시킨다는 아이디어를 바탕으로 하고 있다. 위 그림에서 볼 수 있듯 여러 개의 데이터셋을 사용하더라도 Hidden Node는 공유한다. 이 방법에서 입력 임베딩 벡터는 두 가지로 나뉜다. $x^m_t, x^n_t$ 은 특정태스크에 맞춰진(Task specific) 임베딩 벡터이고, $x^s_t$ 는 서로 공유하는(Shared) 임베딩 벡터이다. 이 두 가지 임베딩 벡터를 동시에 학습시켜 모델의 성능을 향상시키는 것이 이 방법의 목표이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88295150-ff662100-cd37-11ea-9688-0fa60bf53f9b.png" alt="rnn2" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

두 번째는 동일한 임베딩 벡터를 사용하되 Hidden Layer를 2개로 구성하여 각 단게의 Hidden state가 다음 단계에 서로 영향을 주도록 하는 구조이다. 마지막은 Hidden Layer를 3개로 구성하여 각각의 임베딩 벡터가 2개 $(h^m_t, h^s_t) \text{or} (h^n_t, h^s_t)$ 에 영향을 주도록 한다. 그리고 Shared Hidden Layer $h^s_t$ 가 각각의 $h^m_t, h^n_t$ 에게 영향을 주어 성능을 높이고자 하는 방식이다. 아래는 두 번째(Coupled-Layer)와 세 번째(Shared-Layer)의 구조를 도식화한 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88295154-fffeb780-cd37-11ea-936d-626c92b7e8c3.png" alt="rnn3" style="zoom:67%;" /></p>

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88295157-012fe480-cd38-11ea-8694-8fc5f29643f4.png" alt="rnn4" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

## Attention

이전에도 등장했던 어텐션(Attention) 메커니즘은 특정 Task(여기서는 분류)를 수행하는 과정에서 중요한 역할을 하는 단어에 가중치를 부여하는 방식이다. 어텐션 메커니즘의 목적은 $C$ 라는 문맥 벡터(Context Vector)를 만드는 것이다. 일련의 과정을 통해 만들어진 Context Vector, $C$ 는 문서를 분류하는 시점에서 어떤 단어가 더 중요한 지를 알려주는 증거가 된다. $C$ 를 구하는 수식은 아래와 같으며 아래 식에서 $\alpha$ 는 가중치이고, $h$ 는 은닉 상태 벡터(Hidden state vector)이다.

 
$$
C = \sum^T_{i=1} \alpha_ih_i
$$



$\alpha$ 를 계산하기 위해 제안된 2가지 어텐션 메커니즘이 있다. 두 방법 모두 2015년에 제안된 방식인데 첫 번째는 Bahdanau가 제안하였고, 두 번째는 Luong이 제안하였다. Bahdanau의 방식은 가중치(Attention score)를 따로 학습시키는 방식이다. Luong의 방식은 별도의 가중치 학습 없이 현재 Hidden state와 Target state의 벡터 연산(예를 들면, 내적)을 통해서 구한다. 실험 결과 두 방법 사이에 유의미한 성능 차이기 없었기 때문에 좀 더 단순한 Luong의 것을 사용하는 것이 일반적이 되었다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88295160-01c87b00-cd38-11ea-9cc0-3d389fdf66de.png" alt="rnn5" style="zoom: 80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

그렇다면 Luong 어텐션에 대해서 좀 더 자세히 알아보자. 일반적인 RNN에서는 해당 토큰 까지의 은닉 상태 벡터(Hidden state vector)인 $h_t$ 만을 넘겨주어 $\tanh h_t$다. 하지만 어텐션 메커니즘에서는 컨텍스트 벡터인 $c_t$ 와 $h_t$ 를 Concatenate 하여 구하게 된다. 아래는 어텐션을 적용한 은닉 상태 벡터 $\tilde{h_t}$ 를 수식으로 나타낸 것이다.


$$
\tilde{h_t} = \tanh (W_c [c_t,h_t])
$$


이렇게 구해진 벡터에 소프트맥스 함수를 취하여 해당 문장이 특정 클래스에 속할 확률을 구하게 된다. 


$$
p(y_t|y_{y<t}, x) = \text{softmax}(W_s\tilde{h_t})
$$


위 계산을 위해서는 컨텍스트 벡터인 $c_t$ 를 구할 수 있어야 하고 이를 구하기 위해서는 가중치 값인 $\alpha_t$ 를 구할 수 있어야 한다. Luong 어텐션 메커니즘에서는 $\alpha_t$ 를 Hidden state vector $\bar{h_s}$ 와 Target state $h_t$ 벡터의 연산으로 구한다. 이를 수식으로 나타내면 아래와 같다.


$$
\text{score}(h_t, \bar{h_s}) = \begin{cases} h_t^T\bar{h_s} \\ 
h_t^TW_\alpha\bar{h_s} \\ v_a^T \tanh(W_c[c_t;h_t]) \end{cases}
$$


이렇게 구해진 연산값(score)에 소프트맥스 함수를 적용하여 가중치 $\alpha$ 를 구한다.


$$
\alpha_t(s) = \frac{\exp(\text{score}(h_t, \bar{h_s}))}{\sum_{s^\prime}\exp(\text{score}(h_t, \bar{h_s^\prime}))}
$$


이렇게 구해진 가중치와 Hidden state vector $\bar{h_s}$ 를 내적하면 Context vector $c_t$ 를 구할 수 있다.


$$
c_t = \bar{h_s} \alpha_t
$$


## RNN for Document Classification

2016년에는 어텐션 메커니즘을 계층적으로 쌓아 문서를 분류하는 방법이 제시되었다. 계층적 어텐션 네트워크(Hierarchical Attention Network)는 문장에 대해 단어 단위로 RNN을 수행하여 분류를 진행한 뒤, 그 결과 다시 RNN에 입력하여 문서를 분류하는 방법이다. 계층적 어텐션 네트워크를 도식화하여 나타내면 아래와 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88295162-02611180-cd38-11ea-807e-c5b3f64b7c0c.png" alt="rnn6" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

첫 번째 단계에서는 단어 시퀀스를 양방향(Bi-directional)으로 인코딩하며 두 번째 단계에서는 단어 단계에서 어텐션을 적용한다. 문장 내 단어 단위의 어텐션이 끝나면 문서 내 문장의 시퀀스를 인코딩 하게 된다.(세 번째 단계) 이렇게 인코딩된 벡터에 문장 단계의 어텐션을 적용하여 마지막으로 문서를 분류하게 된다.(네 번째 단계)

논문에서는 첫 번째와 세 번째 단계에 사용되는 인코더로 Bi-directional GRU를 사용하였다. 아래는 GRU의 Hidden Node를 도식화하여 나타낸 것이다. 

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88295164-02f9a800-cd38-11ea-821a-269b6220854b.png" alt="rnn7" style="zoom: 50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

이 인코더는 정방향과 역방향의 Hidden state vector $\overrightarrow{h_t}, \overleftarrow{h_t}$ 를 만들어내고 두 벡터를 Concatenate하여 $h_t$ 를 만들게 된다. 이렇게 구해진 $h_t$ 는 활성화 함수 $\tanh$ 를 거친 뒤 어텐션 $\alpha_t$ 을 적용하여 문장의 컨텍스트 벡터 $s$ 를 만든다.

이렇게 구해진 $s_i$ ( $i$ 번째 문장에서의 $s$ )에 다시 양방향 GRU를 적용하여 $\overrightarrow{h_i}, \overleftarrow{h_i}$ 를 구하고 두 벡터를 Concatenate하여 $h_i$ 를 만든다. 이렇게 구해진 $h_i$ 는 활성화 함수 $\tanh$ 를 거친 뒤 어텐션 $\alpha_i$ 을 적용하여 문장의 컨텍스트 벡터 $v$ 를 만든다. 이 $v$ 를 완전 연결층에 통과시킨 후 소프트맥스를 취하여 문서가 특정 클래스에 구할 확률을 구해낸다.



## CNN vs RNN for Classfication

아래는 Localization을 통해 CNN과 RNN에서 긍정과 부정으로 문서를 분류하는 데 큰 영향을 끼친 단어들을 시각화한 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88295912-fde92880-cd38-11ea-8436-fbbfa80b687f.png" alt="rnn8" style="zoom:67%;" /></p>

![rnn9](https://user-images.githubusercontent.com/45377884/88295957-0f323500-cd39-11ea-8879-5d1a4c674cf8.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

위 결과를 보면 CNN은 영향을 미치는 구(Phrase)를 좀 더 잘 판별하고 있음을 알 수 있다. CNN은 Window 크기에 따라 한 번에 수 개의 단어를 탐지하므로 이렇게 판단한다는 것을 알 수 있다. 반면에 RNN은 단어 하나씩을 듬성듬성 골라내고 있는 것을 알 수 있다.