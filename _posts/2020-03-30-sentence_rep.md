---
layout: post
title: Text Classification & Sentence Representation
category: NLP
tag: NLP
---



본 포스트의 내용은 Edwith 조경현 교수님의 강의를 참고하여 작성되었습니다.



## 1) Overview

- 텍스트 분류(Text Classification)
  - 입력 데이터 : 자연어 문장 / 문단
  - 출력 데이터 : 입력 데이터가 속해있는 카테고리
- 예시
    - 감성 분석(Sentiment Analysis) : 긍정적인가? 부정적인가?
    - Text Categorization : 입력 데이터가 어떤 주제에 속해있는가?
    - 의도 파악(Intent Classification) : 입력 데이터가 의도하는 바는 무엇인가?



<br/>

## 2) Tokenization

- 텍스트를 어떻게 표현할 것인가?

  - 텍스트는 길이가 가변적인 토큰들의 연속 : $X = (x_1, x_2, ..., x_T)$
  - 각 토큰들은 하나의 단어에 속한다 : $x_t \in V$
  - 사전으로부터 단어를 찾아 나눈다(Tokenization)

  - 예시
    - (커넥트, 재단에서, 강의, 중, 입니다, .) : 공백을 기준으로 하여 토큰화
    - (커넥트, 재단, 에서, 강의, 중, 입니다, .) : 형태소를 기준으로 하여 토큰화
    - (커, 넥, 트, [], 재, 단, 에, 서, [], 강, 의, [], 중, [], 입, 니, 다, .) : 음절을 기준으로 하여 토큰화
    - 기타 등등



- 하나의 예시 : (커넥트, 재단에서, 강의, 중, 입니다, .) $\rightarrow (5241, 827, 20, 288, 12, 19, 5)$
  - 각각의 단어가 사전의 몇 번째 인덱스에 있는지를 찾아서 숫자로 바꾸어준다. Integer sequence로 변경.



- 각 토큰들을 어떻게 표현할 것인가?

  - 토큰은 각각 인덱스로 표현되어 있음
  - 그것의 의미를 어떻게 우리가 표현할 것인가?
  - **원-핫 인코딩(One-hot Encoding)** : 토큰들 사이의 관계를 모른다고 가정하고 바꾸기

  $$
  x = [0, 0, 0, ..., 0, 1, 0, ..., 0, 0] \in {\{0,1\}}^{\vert{V}\vert} \\
  \\ {\quad} \\ 
  \vert V \vert :\text{ 단어의 개수} \\
  \sum^{\vert V \vert}_{i=1}x_i = 1 : \text{ 오직 하나의 원소만 1 (나머지는 0)}
  $$

  - 원-핫 인코딩의 장점(이자 단점) : 어떤 두 토큰을 뽑더라도 서로의 거리가 같음. 단어간의 Arbitrary(임의적인) 것을 잘 담아내지만 우리가 원하는 것과는 다르다.

  $$
  \vert\vert x-y \vert\vert = c > 0, \text{ if} \quad x \neq y
  $$



- 우리가 원하는 것은 각 토큰의 의미를 뽑아내는 것 (단어, 즉 토큰 간의 거리가 달라져야 함)
  - **Table Lookup** : 원-핫 인코딩 된 토큰에게 벡터를 부여하는 과정. 위에서 나온 것처럼 원-핫 인코딩 벡터 $(x)$ 와 가중치 벡터의 연속 벡터 공간 $(W)$ 을 내적한다.
  - Table Lookup Operation 이후에는 입력 텍스트는 연속적인 고차원 벡터 시퀀스가 된다.
  
  $$
  \mathbf{X} = (e_1, e_2, ..., e_T),  \text{ where } e_t \in \mathbb{R}^d
  $$
  
  

<br/>

## 3) CBoW & RN & CNN

- **CBoW(Continuous Bag-of-Words)** 

  - 토큰들의 순서를 무시 : $(x_1, x_2, ..., x_T) \rightarrow \{x_1, x_2, ..., x_T \}$ 
  - 단순히 토큰 벡터의 평균만을 계산한다. 그 평균이 전체 문장을 의미하는 것으로 간주

  $$
  \frac{1}{T} \sum^T_{t=1}e^t : \text{ differentiable operator, 미분 연산자} \\
  \text{DAG(유향 비순환 그래프) 내에서 하나의 operator node로 존재}
  $$

  - **N-gram** 을 통한 일반화
    - N-gram : 토큰을 몇 개씩 묶어서 볼 것인가? (Uni-gram, Bi-gram, Tri-gram)

  - 순서를 무시함에도 텍스트 분류에 있어서는 **매우 성능이 좋다!** (Text Classification 문제에 있어서 BaseLine Model로 삼는다.)
  - **FastText** 로 쉽게 해볼 수 있다.



- CBoW 기반의 Multi-class Text Classifier (다중 클래스 분류기)

$$
\left[{\begin{array}{ccc} x_1 \\ x_2 \\ ... \\ x_T  \end{array}} \right] \cdot W : \text{ Table Lookup } \Rightarrow \text{ Average } \Rightarrow \text{ An arbitrary sub-graph }(\theta) \Rightarrow \text{Softmax Classifier}
$$



- **RN(Relation Network)** : Skip Bi-grams

  - 모든 단어(토큰)를 쌍(pair)으로 묶어낸다고 하자 : $(x_i, x_j), \forall {i \neq j}$
  - 각 쌍에 대하여 두 토큰 벡터를 조합

  $$
  f(x_i, x_j) = W \phi (U_{\text{left}}e_i + U_{\text{right}}e_j) \\
  \phi : \text{ 각 원소에 대한 비선형 함수, ex) tanh or ReLU}
  $$
  - 각 쌍마다의 관계를 계산하여 평균을 낸다.

  $$
  \text{(관계도)} \quad RN(X) = \frac{1}{2N(N-1)}\sum^{T-1}_{i=1}\sum^{T}_{j=i+1}f(x_i, x_j)
  $$



- RN 기반의 Multi-class Text Classifier (다중 클래스 분류기) : $\phi : \text{An Arbitrary sub-graph}$

$$
\text{from } \left[{\begin{array}{ccc} x_1 \\ x_2 \\ ... \\ x_T  \end{array}} \right] \Rightarrow \left[{\begin{array}{ccc} RN_{12}, RN_{13}, ..., RN_{1T} \\ RN_{21}, RN_{23}, ..., RN_{2T} \\ RN_{31}, RN_{32}, ..., RN_{3T}\\ ... \\ RN_{T1}, RN_{T2}, ..., RN_{TT-1}  \end{array}} \right] \Rightarrow \text{Average} \Rightarrow \phi \Rightarrow \text{ Softmax Classifier}
$$



- **CNN(Convolutional Neural Network)** : Local의 패턴을 인식

  - k-gram의 계층을 파악한다.
  - 자연어의 경우에는 1차원 Conv Layer를 사용 (이미지는 2차원) : 모든 k-grams를 고려

  $$
  h_t = \phi (\sum^{k/2}_{\tau=-k/2} W_{\tau}e_{(t+\tau)}), \text{ resulting in } H = (h_1, h_2, ..., h_T)
  $$

  - Conv Layer를 쌓아갈수록 점점 Window가 커지게 된다.
  - $ \text{Tokens } \rightarrow \text{ Multi-word expressions } \rightarrow \text{ Phrases } \rightarrow \text{Sentence} $



- 다양한 기법들이 이미 연구되어 있다
  - Multi-width Convolutional Layers, Dilated Convolutional Layers, Gated Convolutional Layers and so on. 

<br/>

## 4) Self Attention & RNN

- ?



<br/>