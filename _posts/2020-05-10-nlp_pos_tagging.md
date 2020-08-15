---
layout: post
title: 품사 태깅 (Part-of-Speech Tagging, POS Tagging)
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# POS Tagging

이번 게시물에서는 품사 태깅에 대해 좀 더 자세히 알아봅니다. 품사 태깅의 개괄적인 정보를 포함하여 어휘 분석과 구조 분석에 대한 내용은 [이곳](https://yngie-c.github.io/nlp/2020/05/09/nlp_lexical_syntax_analysis/) 에서 보실 수 있습니다.

## Pointwise prediction

품사 태깅 알고리즘은 크게 3가지로 나눌 수 있습니다. 첫 번째는 **점별 예측(Pointwise prediction) 모델** 입니다. 이 방식은 해당 단어를 포함한 단어 시퀀스를 분류기에 넣으면 해당하는 단어의 품사를 예측하는 알고리즘입니다. 점별 예측 알고리즘의 예로는 최대 엔트로피 모델(Maximum Entropy Model) 과 서포트 벡터 머신이 있습니다. 각 단어의 품사를 따로 예측하기 때문에 Pointwise라는 이름이 붙었습니다. 아래는 점별 예측 모델을 통해 품사 태깅을 하는 과정을 도식화한 이미지 입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/89734066-a535c180-da94-11ea-9937-c5d033b1d837.png" alt="pos1" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

이 중 최대 엔트로피 모델이 어떻게 작동하는지 알아보도록 합시다. 최대 엔트로피 모델은 접두사(prefix), 접미사(suffix) 및 주변 단어들로부터 해당 품사에 대한 정보를 얻어내는 방식입니다. 아래 표를 참고하여 쉬운 예시를 들어보겠습니다.

<p align="center"><img src="https://t1.daumcdn.net/cfile/tistory/2705994D590C9E0B15" alt="tagging_table" style="zoom: 80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://t1.daumcdn.net/cfile/tistory/2705994D590C9E0B15">t1.daumcdn.net</a></p>

만약 *"playing"* 이라는 단어의 품사를 매길 때, 해당 단어의 접미사가 *"-ing"* 이므로 위 표에서 VBG 를 할당하는 방식입니다. 아래는 최대 엔트로피 모델을 수식으로 표현한 것입니다. $p(t \vert C)$ 는 특정 문맥이 주어졌을 때 각 품사가 태깅될 확률입니다.



$$
p(t|C) = \frac{1}{Z(C)} \exp(\sum^n_{i=1} \lambda_if_i(C,t)) \\ p(t_1, \cdots , t_n | w_1, \cdots , w_n) \approx \prod^n_{i=1} p(t_i|w_i)
$$



위 식에서 $\lambda_i$ 는 가중치를 나타냅니다. 더 중요한 정보에 대해서 더 큰 값을 나타내게 됩니다. $Z(C)$ 는 적절한 확률 분포를 위한 정규화 상수(Normalization constant)입니다. 



## Probabilistic prediction

두 번째 분류는 **확률 기반의 모델(Probabilistic models)** 입니다. 확률 기반의 모델은 점별 예측 모델과 달리 문장 전체를 한꺼번에 입력받는다는 특징을 가지고 있습니다. 그리고 입력 문장에 하여 가장 확률이 높은 품사 조합을 할당하게 됩니다. 입력 문장을 $X$ , 할당되는 품사를 $Y$ 라고 하였을 때 확률 기반의 모델을 아래와 같은 수식으로 나타낼 수 있습니다.


$$
\text{argmax}_{Y}P(Y|X)
$$


확률 기반의 모델 안에도 두 가지 소분류가 존재합니다. 히든 마르코프 모델(Hidden Markov model, HMM)과 같이 주어진 문자에 대해 가장 적절한 태그를 찾는 Generative sequence models 과 Conditional Random Field(CRF)와 같이 분류기로 전체 문장의 시퀀스를 예측하는 Discriminative sequence models이 있습니다.

이 중 Generative sequence model 에 대해 먼저 알아봅시다. 이 알고리즘은 베이즈 확률을 이용합니다. 첫 번째 토큰에 대한 태그를 먼저 달고 이후에 올 토큰에 대해서는 조건부 확률을 활용하여 품사를 할당하게 됩니다. 베이즈 룰(Bayes' Rule)을 활용하여 확률 기반 모델의 수식을 아래와 같이 변형할 수 있습니다.


$$
\text{argmax}_{Y}P(Y|X) = \text{argmax}_{Y} \frac{P(X|Y)P(Y)}{P(X)} = \text{argmax}_{Y} {P(X|Y)P(Y)}
$$


위 식에서 $P(X \vert Y)$ 는 단어와 품사 사이의 상관관계를 나타내는 Emission probabilities, $P(Y)$ 는 품사와 품사 사시의 상관 관계를 나타내는 Transition probabilities 입니다. 대표적인 Generative sequence model인 은닉 마르코프 모델(Hidden Markov Models, HMMs) 마르코프 가정, 즉 모든 토큰이 독립이라는 가정을 추가하여 $P(X \vert Y)$ 와 $P(Y)$ 를 구해냅니다. 수식으로는 아래와 같이 나타낼 수 있습니다.


$$
P(X | Y) \approx \prod^{l}_{1} P_E(x_i|y_i) \qquad
P(Y) \approx \prod^{l+1}_{i=1} P_T(y_i|y_{i-1})
$$


아래는 은닉 마르코프 모델을 사용하여 품사 태깅을 진행하는 과정을 이미지로 나타낸 것입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/89734067-a666ee80-da94-11ea-932d-3fd4d64ed934.png" alt="pos2" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

다음으로는 Discriminative sequence model 중 대표적인 알고리즘인 Conditional Random Field(CRF)에 대해 알아봅시다. 이 방법은 성능이 좋기 때문에 트랜스포머를 활용한 Pretrained model에서도 가장 많이 사용되는 품사 태깅 알고리즘 입니다. Generative 방식은 $P(Y)$ 를 통해서 이전 토큰의 품사가 다음 토큰의 품사를 결정하는 데에 영향을 끼쳤습니다. 하지만 Discriminative 방식은 이전 시퀀스의 품사가 주는 영향을 제한하여 완화하였습니다. 그리고 연속적으로 품사를 태깅하는 것이 아니라 모든 문장을 보고 난 이후에 한번에 태깅합니다. 아래는 Generative와 Discriminative 방식을 비교한 이미지입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/89734068-a6ff8500-da94-11ea-896d-5b7dce69382d.png" alt="pos3" style="zoom:150%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf">people.cs.umass.edu</a></p>



## Neural Network based Models

신경망을 기반으로 품사를 태깅하는 모델(Neural Network based models)도 있습니다. 신경망 기반의 모델도 크게 두 가지로 나눌 수 있습니다. 첫 번째는 Window 기반의 모델입니다. 이 모델은 특정 크기의 Window를 선정하여 타겟 단어 주위의 단어로부터 타겟 단어의 품사를 예측합니다. 아래는 Window 기반의 모델을 도식화하여 나타낸 것입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/89735051-be417100-da9a-11ea-8d23-273c8f076966.png" alt="windowBased" style="zoom:120%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://jmlr.csail.mit.edu/papers/volume12/collobert11a/collobert11a.pdf">Natural Language Processing (almost) from Scratch by Collobert</a></p>

두 번째는 Sentence 기반의 모델입니다. 문장 내 단어들을 특성화한 뒤 문장을 특성으로 표현하여 신경망 모델에 넣어 품사를 태깅하게 됩니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/89735050-bd104400-da9a-11ea-82da-29c2f2641ef8.png" alt="sentenceBased" style="zoom:120%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://jmlr.csail.mit.edu/papers/volume12/collobert11a/collobert11a.pdf">Natural Language Processing (almost) from Scratch by Collobert</a></p>

신경망 모델 중에서는 RNN 가장 많이 쓰입니다. RNN에 대한 일반적인 설명은 [이곳]([https://yngie-c.github.io/deep%20learning/2020/06/25/dl_rnn/](https://yngie-c.github.io/deep learning/2020/06/25/dl_rnn/)) 에서 볼 수 있습니다. 아래는 RNN을 사용하여 각 단어의 품사를 태깅하는 과정을 이미지로 나타낸 것입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/89735285-5855e900-da9c-11ea-9a83-82108dc9b89a.png" alt="rnnPos" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>



최근에는 다양한 모델을 혼합하여 사용하기도 합니다. 아래는 장단기 기억망(LSTM)과 합성공 신경망(ConvNet), 그리고 CRF를 혼합하여 사용한 모델을 도식화한 것입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/89735415-580a1d80-da9d-11ea-9b3a-c9f5e6c79bb2.png" alt="hybridPOS" style="zoom:100%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://arxiv.org/pdf/1603.01354.pdf">End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF by Ma and Hovy</a></p>