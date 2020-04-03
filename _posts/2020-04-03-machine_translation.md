---
layout: post
title: Machine Translation
category: NLP
tag: NLP
---



본 포스트의 내용은 Edwith 조경현 교수님의 강의를 참고하여 작성되었습니다. 책은 [밑바닥부터 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) 를 참고하였습니다.



# 1) Overview

## Machine Translation

- Input : Source Language $L_s$

- Output : Target 언어에 맞는 적절한 Sentence $L_T$

- 문제

  - Supervised Learning : 특정 문장이 주어지면 그것의 번역을 내놓는것
  - 모든 가능한 번역 중 조건부 확률분포를 계산해낸다

  $$
  p(Y=(y_1, ..., y_T) \vert X = (x_1, ...,x_{T^\prime}))
  $$

  

<br/>

# 2) Encoder & Decoder

## Token Representation

- One-hot Vectors

  - 각 단어들을 유니크한 토큰으로 만든다
    1. Tokenize : 구두점을 기준으로 분리하고 모든 구두점 변경하기 (Spacy.io, NLTK, Moses' tokenizer)
    2. Subword Segmentation : 더 작은 단위로 나눈다. Ex) *"going"* $\rightarrow$ *"go", "ing"*
    3. 모든 유니크한 단어들을 모은 뒤, 그것들을 Frequency에 따라 정렬하고 각각 Indices를 할당
  - 그것들을 모두 원-핫 벡터로 나타낸다.

  



## Encoder

- Source sentence를 Sentence Representation Vector들의 집합으로 변환

  - 인코딩된 벡터들의 수는 문장의 길이에 비례한다.

  $$
  H = (h_1, h_2, ...,h_{T^\prime})
  $$

  - RNN이 널리 쓰이지만, CNN이나 Self-Attention의 사용도 점점 늘어나는 추세



- 꼭 Fixed size single 벡터로 압축할 필요없음
  - Collapsing은 정보 손실을 의미하기 때문이다.
  - Fixed size 벡터로 나타낸 경우 문장의 길이가 늘어나면 전체 source sentence를 인코딩 하는 것이 어려워짐. (앞부분은 인식을 잘 하지만 문장 뒤에는 Translation이 아닌 Generation을 하게 됨)
  - 벡터의 크기가 정해져 있으면 시스템이 긴 문장을 번역하지 못하게 됨



## Decoder - Language Modeling

- Autoregressive Language Modelling with an inifinite context $(n \rightarrow \\infin)$

  - 일관성있는 문장을 생성해내려면 더 큰 Context가 필요
    - Semantic(의미적인 것)은 Source sentence로부터 제공될 수 있지만 Syntactic(통사적, 문법적인 것)은 언어 모델에 직접적으로 영향을 받음
  - Recurrent Networks, Self-Attention and (Dilated) Convolutional Networks
    - Casual structure가 필요.

  

- Conditional Language Modelling

  - Supervised Learning

  $$
  p(Y \vert X) = \prod^T_{t=1}p(y_t \vert y_{<t}, X) \\
  \text{Input - } X : \text{전체 source sentence, } y_{<t} : \text{선행하는 토큰} \\
  \text{Output - }y_{t} : \text{타깃 토큰} \qquad \qquad \qquad \qquad \qquad \qquad \quad
  $$

$$
\text{Matmul}
\left[{\begin{array}{ccc} x_1 \\ x_2 \\ ... \\ x_{t^\prime}  \end{array}} \right] \left[{\begin{array}{ccc} y_1 \\ y_2 \\ ... \\ y_{t-1}  \end{array}} \right]^T
\Rightarrow \text{Softmax function} \Rightarrow p(y_t \vert y_{<t}, X)
$$



<br/>



# 3) RNN Neural Machine Translation

## Operation

1. Source sentence Representation

- Bidirectional(양방향) Recurrent Network
- 각 위치에서 추출된 벡터는 Context-dependent vector representation 



2. Target Prefix Representation

- Unidirectional(단방향) Recurrent Network
- Target prefix 압축하기

$$
z_t = \text{RNN}_{\text{decoder}}(z_{t-1}, y_{t-1})
$$



3. Attention Mechanism

- Source sentence의 어느 부분이 다음 토큰을 예측하는데 관련되어 있는가?

$$
\sum \text{RN(Relational Net)} \times \text{Weighting factor} (\alpha) = c_t
$$

- Time-dependent Source context vector $c_t$



4. Source context vector 와 Target prefix vector를 결합해준다.

- $z_t$ 와 $c_t$ Single 벡터로 조합한다.



## Conceptual process

1. Encode : 어떤 것을 번역해야 할 지 전체 문장을 읽는다. (Bidirectional RNN)
2. Attension : 각 스텝마다 어떤 토큰이 다음 단어 번역을 위해서 중요한 역할을 하는지 결정한다.
3. Decode : 어떤 것이 번역되어 온 지를 바탕으로 다음 타깃 토큰을 정한다.
4. Repeat : \<eos> 가 나올 때까지 1-2-3의 과정을 반복한다.

