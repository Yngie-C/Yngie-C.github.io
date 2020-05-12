---
layout: post
title: 전처리 (Preprocessing)
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# 어휘 분석 (Lexical Analysis)

텍스트 데이터 전처리에서 **어휘 분석(Lexical Analysis)** 의 목적은 문자(Character)의 시퀀스(Sequence)를 토큰(Token)의 시퀀스로 바꾸는 것이다. 자연어처리(Natural Language Processing, NLP)에서는 기본 단위로 형태소(morpheme)를 사용하고 텍스트마이닝(Text mining)에서는 대부분 기본 단위로 단어를 사용한다. 어휘 분석은 주로 토큰화, POS(Part-of-Speech) Tagging으로 이루어지며 추가적으로 개체명 인식(Named Entity Recognition, NER), 명사구 인식, 문장 분절 등이 수행된다.



## 문장 분절 (Sentence splitting)

가장 먼저 **문장 분절(Sentence splitting)** 을 수행한다. 문장을 구분하는 것은 자연어 처리에서는 매우 중요하지만 Topic Modeling 등 텍스트 마이닝의 특정 분야에서는 중요하지 않기도 한다. 문장과 문장을 나누는 경계는 일반적으로 *"."(마침표), "?"(물음표), "!"(느낌표)* 와 같은 문장 부호에 의해 결정된다. 하지만 *"Mr.Lee", "3.141592", "Samsung Corp."* 등 문장이 끝나지 않는 경우에 사용되는 경우도 있다. 특정 어구를 인용한 경우 인용문이 문장 내에 존재하는 복잡한 케이스도 존재한다. 이후에 나올 분석 과정을 진행하기 위해서 문장을 정확히 구분하는 것은 필수적이다. 일반적으로 룰베이스 알고리즘이나 분리된 문장을 모델에 학습시키는 방법을 사용한다.



## 토큰화 (Tokenization)

두 번째로 실시되는 어휘 분석은 **토큰화(Tokenization)** 이다. 토큰화란 텍스트를 **토큰(Token)** 이라는 기본적인 단위로 나누는 과정이다. 일반적으로 공개된 분석기(Tokenizer)를 사용하며 각 분석기마다 도출되는 토큰이 다르다. 아래는 MC tokenizer와 SCAN tokenizer로 특정 문서를 토큰화한 결과다. 자신이 필요한 토큰에 맞게 분석기를 사용하는 것이 중요하다.

![tokenizer](https://user-images.githubusercontent.com/45377884/81628131-cc033e00-943a-11ea-8582-303e3d175b92.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

문장 분절과 마찬가지로 토큰화도 완벽하게 하기 어렵다. 단어 내에 *"-"(하이픈)* 이 있는 경우도 있고 *"C++", "A/C", ":-)", "ㅋㅋㅋㅋㅋㅋㅋㅋ"* 등 토큰을 구분하기 어려운 특수한 경우가 많기 때문이다. 더구나 몇몇 언어는 중국어처럼 공백이 존재하지 않아 토큰을 나누기 어려운 경우도 있다.



## 형태 분석 (Morphological analysis)

세 번째는 **형태 분석(Morphological analysis)** 이다. 형태 분석의 목적은 의미를 크게 훼손하지 않는 선에서 단어를 Normalize 하여 텍스트 데이터의 차원을 줄이기 위함이다. **Stemming** 과 **Lemmatization** 의 두 가지 방법이 있다.

먼저 Stemming은 정보 추출 관점에서 자주 사용된다. 룰베이스 알고리즘을 사용하여 단어들의 기본 형태를 찾으며 대개 접사를 잘라내는 데 기반을 두고 있다. 영어의 경우 기본적인 알고리즘은 Porter Stemmer를 사용한다. 단어에 상관없이 기본 형태로 잘라내므로 간단하고 빠르다는 장점을 가지고 있다. 하지만 룰베이스 특성상 언어 종속적이며, 없는 단어가 생성된다. 또 서로 다른 단어가 하나의 기본형으로 귀결되는 경우도 있기 때문에 의미(Semantic) 관점에서는 문제가 발생할 수도 있다.

 Lemmatization은 품사를 보정한다. 실제 단어인 Lemma 로 잘라내기 때문에 의미 분석에서 유리하며 만들어진 사전을 기반으로 분석을 수행하기 때문에 오류가 적다는 장점이 있다. 하지만 Stemming에 비해 복잡하고 속도가 느리며 언어에 따라 추가적으로 해주어야 될 일이 많아지는 단점이 있다. Stemming과 Lemmatization 장점과 단점이 있다. 정보 검색에는 전자를 더 자주 사용하지만 대부분의 텍스트 마이닝에서는 의미를 중요시하기 때문에 후자를 더 많이 사용한다. 아래는 두 방법을 비교한 예시이다.

<img src="https://user-images.githubusercontent.com/45377884/81628366-767b6100-943b-11ea-9ae9-955f403e3b8a.png" alt="morpho" style="zoom: 50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>



## POS Tagging

다음으로 **품사 태깅(Part-of-Speech tagging, POS tagging)** 을 사용한다. 문장이 주어졌을 때 토큰에 해당하는 품사를 찾아주기 위해서 POS tagging을 실시한다. POS tagger가 토큰을 스캔한 뒤에 순차적으로 품사를 지정한다. 토큰화에서 분석기마다 다른 결과를 내놓듯이 POS tagging도 tagger의 종류에 따라 결과가 달라진다. 머신 러닝으로 POS tagging을 수행하기 위해서 훈련 데이터로 수동으로 품사를 매긴 말뭉치가 필요하다. 훈련 데이터와 비슷한 도메인의 테스트 데이터에 대해서는 잘 작동하는 편이지만 도메인이 달라지면 성능에 차이가 생긴다. 의사결정트리, 은닉 마르코프 모델, 서포트 벡터 머신 등 다양한 알고리즘을 사용할 수 있다. 최근에는 Brill tagger 와 같은 트랜스포메이션 기반의 tagger 가 많이 사용된다.

그 중 Pointwise prediction 은 개별 단어 시퀀스를 최대 엔트로피 모델이나 서포트 벡터 머신과 같은 분류기에 넣어 품사를 도출해내는 방식이다. 분류기가 목표 단어 앞뒤 시퀀스를 탐색한 뒤 품사를 할당한다. 최대 엔트로피 모델(Maximum Entropy Model)을 살펴보자. 먼저, 태그 예측을 위해서 특성을 인코딩한다. 그다음 아래의 태깅 모델을 사용하여 토큰마다 품사를 할당한다. 아래 수식에서 $f_i$ 는 특성 값이고 $\lambda_i$ 는 가중치를 나타내는 상수이다.



$$
p(t|C) = \frac{1}{Z(C)} \exp(\sum^n_{i=1} \lambda_if_i(C,t)) \quad p(t_1, \cdots , t_i) \approx \prod^n_{i=1} p(t_i|w_i)
$$



Pointwise와 달리 확률론적 모델(Probabilistic Model)은 문장 전체를 한꺼번에 입력받는다. 그리고 입력 문장에 대하여 가장 확률이 높은 품사 조합을 할당하는 방식이다. $X$ 가 입력 문장이고, $Y$ 가 할당되는 품사일 때 확률론적 모델을 아래와 같은 수식으로 나타낼 수 있다.



$$
\text{argmax}_{Y}P(Y|X)
$$



확률론 모델 중 Generative sequence 모델은 베이즈 확률을 이용한다. 첫 번째 토큰에 대한 태그를 먼저 달고 이후에 올 토큰에 대하여 조건부 확률로 품사를 할당하게 된다. 베이즈 룰(Bayes' Rule)에 따라서 확률론 모델의 수식을 다음과 같이 고쳐쓸 수 있다.



$$
\text{argmax}_{Y}P(Y|X) = \text{argmax}_{Y} \frac{P(X|Y)P(Y)}{P(X)} = \text{argmax}_{Y} {P(X|Y)P(Y)} \\
P(X|Y) : \text{단어와 품사 사이의 상관관계를 나타냄 (Emission probabilities)} \\
P(Y) : \text{품사와 품사 사이의 상관관계를 나타냄 (Transition probabilities)}
$$



Generative sequence 모델 중 하나인 은닉 마르코프 모델(Hidden Markov Models, HMMs)는 독립임을 가정하여 $P(X \vert Y)$ 와 $P(Y)$ 를 구해낼 수 있다. 수식으로 나타내면 다음과 같다.



$$
P(Y) \approx \prod^{l+1}_{i=1} P_T(y_i|y_{i-1}) \\
P(X) \approx \prod^{l}_{1} P_E(x_i|y_i)
$$



Discriminative sequence model은 모든 토큰 조합을 한꺼번에 예측한다. Discriminative sequence model 중 하나인 Conditional Random Field(CRF)는 성능이 좋아 트랜스포머 모델에도 가장 많이 사용되는 POS 방식이다. 

신경망 기반의 POS 모델도 있다. 신경망 기반의 모델은 크게 문장의 일부를 사용하는 윈도우(window) 기반과 문장 전체를 사용하는 문장(sentence) 기반의 모델로 나누어진다. RNN이 가장 잘 맞는 방법론이며 이를 변형한 RNN, LSTM, GRU 를 많이 사용하였다. 최근에는 여러 모델을 결합한 하이브리드 모델을 사용하여 퍼포먼스를 높이려고 하고 있다. 



## 개체명 인식 (Named Entity Recognition, NER)

추가적으로 수행되는 작업 중 하나로 **개체명 인식(Named Entity Recognition, NER)** 이다. 문장의 일부분을 미리 구성한 범주 중 하나로 이름 붙인다. 얘를 들면, *"Get me a flight from New York City to San Francisco for next Thursday."* 에서 *"New York City"* 를 출발 장소로, *"San Francisco"* 는 목적지로, *"Thursday"* 는 출발 시각으로 미리 분류한다.

미리 객체에 대한 사전을 구축한 뒤 룰베이스 알고리즘을 사용한다. 사전 방식 중 하나인 List lookup은 미리 만들어 놓은 리스트에서 같은 개체를 매치시키는 방식이다. 단순하고 빠르다는 장점이 있지만 리스트를 관리하거나 업데이트 하기가 어렵다. 룰베이스 알고리즘인 Shallow Parsing Approach는 특정 구조를 발견하면 구조에 의해서 개체명을 할당하는 방식이다. 예를 들어 *Cap word + Street, Boulevard, Avenue, Crescent, Road* 이 있을 경우 Location 임을 추정하여 할당하게 된다. 모델 기반의 알고리즘인 MITIE나 CRF++, CNN 등도 경우에 따라 사용되곤 한다. 



# Syntax Analysis

**구조 분석(Syntax analysis)** 은 일련의 문장이 들어왔을 때 문법 형식에 맞도록 구조를 분석하는 과정이다. 파서(Parser) 알고리즘을 통해서 구조를 분석한다. 파서는 두 가지 속성을 가지고 있다. Top-down 인지 Bottom-up 인지를 결정하는 방향성(Directionality)과 트리를 깊이를 옆으로 탐색할지 아래쪽으로 파고들지를 결정하는 탐색 전략(Search strategy)가 있다. 이 두 특성에 따라 파서의 알고리즘이 달라진다. 

구문 분석을 표현하는 방식은 트리 방식과 리스트 방식이 있다. 각각의 방식으로 *"John ate the apple"* 을 분석할 경우 아래와 같이 나타낼 수 있다.

<img src="https://user-images.githubusercontent.com/45377884/81628477-b9d5cf80-943b-11ea-999e-316ff531f9ef.png" alt="syntax_rep" style="zoom: 50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

구조 분석을 통해서 항상 하나의 트리만 생성되는 것은 아니다. 언어의 모호성 때문이다. 언어의 모호성은 어휘의 모호성(Lexical ambiguity)과 구조적 모호성(Structural ambiguity)이 있다. *"Time flies like an arrow"* 라는 문장을 구조 분석하면 다음과 같이 어휘의 모호성이 발생한다.

<img src="https://user-images.githubusercontent.com/45377884/81628534-e12c9c80-943b-11ea-91d6-5a7a550d767d.png" alt="lex_ambi" style="zoom:50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

구조적 모호성에 따라서도 분석 결과가 다르게 나타날 수 있다. 다음은 *"John saw Mary in the park"* 문장을 구조분석한 결과를 트리로 표현한 것이다. John이 Mary를 공원 안에서 보았는지 공원 밖에서 보았는지에 따라 구조 분석 결과가 달라지는 것을 볼 수 있다.

<img src="https://user-images.githubusercontent.com/45377884/81628583-00c3c500-943c-11ea-9a9d-f3f2190f9f64.png" alt="str_ambi" style="zoom:50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>