---
layout: post
title: 감성 분석 (Sentiment Analysis)
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# Sentiment Analysis

**감성 분석(Sentiment Analysis)** 이란 텍스트에 들어있는 의견이나 감성, 평가, 태도 등의 주관적인 정보를 컴퓨터를 통해 분석하는 과정이다. 감성 분석은 오래 전부터 연구되어온 Task이지만 언어가 가지고있는 모호성 때문에 쉽지 않은 것이 사실이다. 아래의 예시를 보자.

*"Honda Accords and Toyota Camrys are nice sedans."*

이 문장은 혼다와 도요타 각각 차종에 대해서 긍정을 표현하고 있는 문장이다. 하지만 약간의 내용만 덧붙이더라도 결과는 달라진다.

*"Honda Accords and Toyota Camrys are nice sedans, but hardly the best car on the road"*

추가적인 설명이 붙은 후로는 이게 긍정을 표하는 문장인지, 부정을 표하는 문장인지 파악하기가 어려워진다. 이런 언어의 모호성은 감성 분석을 어렵게 하는 이유가 된다.

감성 분석은 다양한 곳에 사용된다. 기업에서는 고객 피드백, 콜센터 메시지 등 기업 내부의 데이터를 분석하거나 기업과 관련된 뉴스나 그에 달린 댓글, SNS 데이터 등에 담긴 긍/부정을 판단하는 데에 감성 분석을 사용한다. 개인은 영화를 보기 전에 영화평을 참고하는 것과 같이 특정 제품이나 서비스를 사용할 지를 결정하는 데 감성 분석을 사용할 수 있다. 이외에도 광고의 효율을 높이거나 약이 실제로 사람들에게 효과가 있는 지를 알아보는 데에도 감성 분석을 적용할 수 있다.



## Architecture

감성 분석은 크게 두 가지 단계로 이루어져 있다. 문서에서 어떤 부분에 의견이 담겨있는 지를 결정(Opinion definition)하는 것이 첫 번째 단계이다. 두 번째 단계는 이렇게 모아진 의견을 요약(Opinion summerization)하는 것이다.

의견을 정의하는 첫 번째 단계를 더 자세히 들여다 보도록 하자. 이 단계에서 가장 먼저 해야하는 일은 분석에 필요한 4가지 요소를 찾는 일이다. 4가지 요소는 각각 분석의 대상이 되는 개체(Entity)혹은 그 개체의 특성(Aspect/Feature), 개체에 대한 감성(Sentiment), 의견을 표현하는 주체(Opinion holders), 발화 시점(Time)이다.

의견의 종류에는 일반 의견(Regular opinions)과 비교 의견(Comparative opinions)이 있다. 전자는 하나의 개체(혹은 개체의 특성)에 대한 감성을 표현한 것이다. 일반 의견은  *"이 제품은 좋다"* 와 같이 직접적으로 표현(Direct opinions)했는지, *"이 제품은 잘 작동한다"* 와 같이 간접적으로 표현(Indirect opinions)했는지에 따라 나눌 수 있다.

일반 의견으로 한정한다면 대상(Target)이 되는 개체는 하나이다. 그렇기 때문에 의견의 네 가지 요소를 다음과 같이 나타낼 수 있다.


$$
(g_i, so_{ijk}, h_j, t_k)
$$


$g_i$ 는 분석 대상, $h_i$ 는 발화 주체, $t_k$ 는 발화 시점이다. $so_{ijk}$ 는 $i$ 에 대해서 $k$ 시점에 $j$ 가 표현한 감성이다. 분석 대상은 하나의 개체 혹은 토픽(개체의 특성)이다. 하지만 실제 문서에서는 복합적으로 사용되는 경우가 많아 분석이 쉽지 않다. 이런 경우 의견을 제대로 나타내기 위해서 $g$ 를 개체를 나타내는 $e$ 와 특성(aspect/feature)을 나타내는 $a$ 로 분리하여 나타낸다. 의견을 총 5가지 요소로 나타낼 수 있다. 


$$
(e_i, a_{jl}, so_{ijkl}, h_j, t_k)
$$


이 다섯가지 요소를 활용하면 문장의 구조화가 가능하다. 문장 뿐만 아니라 문서와 같이 긴 텍스트에 담긴 감정을 분류하는 문제도 구조화하여 나타낼 수 있다.

일반적인 감성 분석 시스템의 구조는 아래와 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88989608-ff0ced80-d316-11ea-8e57-76a78d484386.png" alt="sentiment1" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

실제로 문서에 담긴 감성을 분석할 때는 요소의 개수를 줄이기 위해서 *"하나의 문서 내에서는 한 사람이 한 개체에 대해서만 평가한다."* 는 가정이 사용된다. 대부분의 리뷰는 발화자(Opinion holder) 자신이 사용한 제품에 대해서 평가하는 것이므로 이런 가정을 잘 만족한다. 감성 분석에는 Unigram, TF-IDF를 통한 가중치 부여, POS Tagging, Negations 등의 특성 공학이 사용된다.

감성 분석의 방법론은 아래와 같은 것들이 있다. 크게는 어휘 기반(Lexicon-based)의 감성 분석과 머신러닝 기반(ML-based)의 감성 분석으로 나눠볼 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88989668-1cda5280-d317-11ea-8704-6c6555e171c1.png" alt="sentiment2" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

## 어휘 기반(Lexicon-based)의 감성 분석

위 그림에서 살펴본 바와 같이 어휘 기반의 감성 분석 방법은 총 세 가지로 나뉜다. 첫 번째 방법은 모든 단어에 대한 감성 사전을 구축하는 것이다. 감성 사전이란 각 단어가 가지는 긍/부정의 정도를 1부터 -1까지의 점수로 나타낸 것이다. 도메인에 따라서 어휘들이 다르게 나타나기 때문에 모든 도메인에 적용할 수 있는 감성 사전을 구축하는 것은 매우 어렵다는 단점이 있다.

감성이 들어갈 수 있는 품사인 명사, 형용사, 동사 키워드를 추출하여 표현을 결합한 후 이에 맞는 긍/부정 레이블링을 한다. 이렇게 구축된 감성 사전을 통해 개체에 따른 극성 및 긍/부정을 시각화 할 수 있으며 연관어 분석을 통해 어떤 단어들이 긍/부정적인 단어들과 같이 사용되었는 지 알 수 있다.

위와 같이 수동으로 감성 사전을 구축하는 방법 외에 두 가지 방법이 더 있다. 하나는 사전 기반(Dictionary-based)의 접근법이고 나머지 하나는 말뭉치 기반(Corpus-based)의 접근법이다. 사전 기반의 접근법과 이전의 감성 사전 구축법의 차이는 기존에 구축되어 있는 외부 사전을 차용한다는 점이다. 이 방법의 가장 큰 단점 역시 도메인과 문맥(Context)에 있다. 기존 구축된 사전과 다른 도메인에 이 방법을 적용할 경우에는 제대로된 의미 파악이 어려울 수 있다. 예를 들어, 영화 리뷰 데이터에서 *"졸립다"* 라는 단어는 부정적인 의미를 가지고 있지만 침대 상품평 데이터에서 *"졸립다"* 라는 단어는 긍정적인 의미를 가지고 있다. 도메인을 고려하지 않고 외부 사전을 차용한다면 이런 차이를 반영하기 어렵다.

말뭉치 기반의 접근 방법은 해당 말뭉치에 맞는 적절한 감성 어휘를 재구축하는 일이다. 이 경우에는 도메인 의존성에 대한 문제는 해결할 수 있지만 좋은 사전 구축을 위해서 많은 데이터(거대한 말뭉치)가 필요하다는 조건을 가지고 있다. 특정 말뭉치를 분석하는 경우에는 해당 단어가 사용되었을 때 문장이 긍정인지 혹은 부정인지를 t-test 를 통해서 판단한다. 이 때 귀무가설 $h_0$ 은 해당 단어가 사용되었을 때와 그렇지 않을 때의 차이가 없는 것이며 이를 기반으로 t-test 를 진행한다. 



## 머신러닝 기반(ML-based)의 감성 분석

어휘 기반이 아닌 머신러닝 모델을 통해서도 감성 분석을 수행할 수 있다. 그 중 하나는 회귀 분석을 통해 사전을 구축하는 방법이다. 평점 등으로 레이블링 된 데이터에 회귀 분석 모델을 적용하여 감성 사전을 구축한 뒤 교차 분석을 통해 감성 사전으로서의 타당한 성능만 확보한다면 해당 말뭉치에 맞는 감성 사전을 구축함과 동시에 감성 분석을 수행할 수 있다.

준지도 학습(Semi-supervised Learning)을 통해 감성 사전을 구축하는 방법도 있다. 이 방법은 또 두 가지로 나뉘는데 첫 번째가 감성 그래프를 구축하는 방법이다. 고차원 단어를 저차원으로 임베딩 후 임베딩 공간에서 거리를 기반으로 네트워크를 구축한다. 감성이 확실하게 드러나는 수 개의 단어를 미리 레이블링(Pre-labeled sentiment words) 한 뒤 공간상의 위치에 따라 각 단어의 감성 점수를 매기는 방식이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88989671-1e0b7f80-d317-11ea-8cba-1aefdcbac401.png" alt="sentiment4" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

준지도 학습을 통한 두 번째 접근 방법은 자가 학습(Self-Training)이다. 그래프 기반의 감성 분류 방법과 같이 특정 단어는 미리 레이블링이 되어 있다. 이 어휘로만 분류기를 학습한 뒤 정답이 없는 어휘에 분류기를 적용하여 결과의 신뢰도가 높으면 정답으로 지정한 후에 분류기를 재학습하는 방식이다. 반복 학습 횟수가 늘어날수록 더 많은 단어가 분류되는 것을 볼 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88989672-1ea41600-d317-11ea-88fc-f9078c7a98ff.png" alt="sentiment5" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

2016년에는 RNTN(Recursive Neural Tensor Network)이라는 방법이 제시되었다. RNTN은 두 가지 벤치마크 모델이 있다. 첫 번째는 RecursiveNN 이다. 각 단어를 아래와 같이 구조적으로 나타낸 후에 구(Phrase)로 결합되면서 그 구의 감성이 어떻게 나타나는지를 학습한다. 발표한 논문에서는 긍정과 부정을 25단계로 구분하였으며 동일한 구에 대하여 3명이 평가한 결과물을 평균내어 레이블링 하였다. 여기서 Recursive라는 단어를 사용하는 이유는 각 단어 혹은 구에 같은 가중치를 적용하기 때문이다. 아래 그림에서는 RecursiveNN이 진행되는 과정을 볼 수 있는데 이 때 사용되는 함수 $g$ 가 $p_1$ 과 $p_2$ 를 나타낼 때 동일하게 사용되는 것을 알 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88989674-1f3cac80-d317-11ea-9b46-23c1db503742.png" alt="sentiment6" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

이 과정을 수식으로 나타내면 다음과 같다. 

$$
y^{\text{word}} = \text{softmax}(W_s \cdot \text{word}) \\
p_1 = f\bigg(W \begin{bmatrix} b \\ c\end{bmatrix} \bigg), \quad p_2 = f\bigg(W \begin{bmatrix} a \\ p_1 \end{bmatrix} \bigg)
$$


위 식에서 함수 $f$ 는 일반적으로 $\tanh$ 가 사용되며 이 때 가중치인 $W$ 는 변하지 않는다.(Recursive) 새로 만들어지는 구를 구성하는 단어 혹은 작은 구를 표현하는 벡터는 Concatenate된 후에 가중치와 내적한다. 

두 번째 벤치마크 모델은 MV-RNN(Matrix-Vector Recursive Neural Network)이다. 이는 기존 RecursiveNN의 한계점을 해결하기 위한 방법이며 더 긴 문장의 문맥을 행렬에 저장하고자 했다.

그리고 이 두 모델을 결합하여 텐서로 쌓아 나타낸 것이 바로 아래의 RNTN이다. RNTN은 이전의 두 벤치마크 모델보다 일반적으로 성능이 더 좋다. 특히 but으로 두 문장이 이어져 있는 경우나 복잡한 부정 표현(High-level Negation)이 있는 경우의 감성을 훨씬 더 잘 판별한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88989692-28c61480-d317-11ea-80f3-fdf6641fce97.png" alt="sentiment7" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>


