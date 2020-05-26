---
layout: post
title: Neural Language Models
category: NLP
tag: NLP
---



본 포스트의 내용은 Edwith 조경현 교수님의 강의를 참고하여 작성되었습니다. 책은 [밑바닥부터 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) 를 참고하였습니다.



# 1) Overview

## Language Modelling

- Input : 문장
- Output : Input 문장의 Score(Probability) 

$$
p(X) = p((x_1, x_2, ... ,x_T))
$$



- Unsupervised Learning 이지만 Supervised Learning 으로 바꿀 것 (Scoring과 Generate는 동일한 Operation)



<br/>

# 2) Autoregressive Language Model

## Autoregressive Sequence Modeling

- 토큰의 distribution이 이전 모든 토큰들에 기반하여 계산

$$
p(X) = p(x_1)p(x_2\vert x_1)p(x_3\vert x_1, x_2) ... p(x_T\vert x_1, x_2 ... x_{T-1}) \\ \text{ }\\ \rightarrow \text{ Conditional distribution}
$$

- 비지도 학습이 지도 학습의 집합이 됨

  - Input : 이전의 모든 토큰 (문장의 부분)
  - Output : 가능한 모든 토큰들의 분포 (Classes)

  $$
  p(\text{Output}\vert \text{Input})
  $$

  - 따라서 Text Classification 문제와 같아짐

$$
p(X) = \prod^T_{t=1} p(x_t \vert x_{<t}) \\
\text{ } \\ \text{Loss Function} : \log{p_\theta (X)} = \sum^N_{n=1}\sum^T_{t=1} \log{p_\theta (x_t \vert x_{<t})}
$$


$$
\left[{\begin{array}{c} x_1 \\ x_2 \\ ... \\ x_T  \end{array}} \right] \cdot W : \text{ Table Lookup } \Rightarrow \text{ Sentence Representation Extractor } \vert \text{ Average } \\ \Rightarrow \text{ An arbitrary sub-graph }(\theta) \Rightarrow \text{Softmax Classifier} \\ \text{ }
$$

- 모든 토큰에 대하여 Text Classification 모델을 Training 시키는 것



## Scoring a sentence

- Ex1) "In Korea, more than half of residents speak Korean."

  - *In* 이 문장 맨 앞에 오는 것이 적절한가(확률이 높은가)?
  - *Korea* 가 In 뒤에 오는 것이 적절한가(확률이 높은가)?
  - *more* 이 In Korea 뒤에 오는 것이 적절한가(확률이 높은가)?
  - *than* 이 In Korea more 뒤에 오는 것이 적절한가(확률이 높은가)?
  - ...
  - *Korean* 이 In Korea more than half of residents speak 뒤에 오는 것이 적절한가(확률이 높은가)?

  

- Ex2) "In Korea, more than half of residents speak **Korean**." vs "In Korea, more than half of residents speak **Finnish**."

  - 전자가 훨씬 가능성이 높다
  - Scoring : NLLLoss(Negative Log Likelihood Loss)



<br/>



# 3) N-gram Language Models

## N-gram

- Autoregressive Language Model 에서는 특정 토큰 이전의 모든 토큰을 보고 Probability를 예측. 
- **N-gram** 모델에서는 이전 모든 단어를 보지않고, 그 단어 이전에 있는 N개의 단어들에 대해서만 Context를 봄.

$$
p(x \vert x_{-N}, x_{-N+1}, ... , x_{-1})
$$

- 조건부 확률(Conditional Probability)과 주변 확률(Marginal Probability)의 정의에 따라 아래와 같이 나타낼 수 있다.

$$
p(x \vert x_{-N}, x_{-N+1}, ... , x_{-1}) = \frac{p(x_{-N}, x_{-N+1}, ... , x_{-1}, x)}{p(x_{-N}, x_{-N+1}, ... , x_{-1})} = \frac{p(x_{-N}, x_{-N+1}, ... , x_{-1}, x)}{\sum_{x \in V}p(x_{-N}, x_{-N+1}, ... , x_{-1}, x)} \\ V : \text{all possible tokens (=vocabulary)}
$$



- Maximum Likelihood[^1] Estimation( **MLE** , 최대 우도 추정)

  - 우도를 측정하기 위한 Data : 위키피디아, 뉴스 기사, 트위터 등등

  - Estimation

    - N-gram의 발생 횟수를 세기 : $p(x_{-N}, x_{-N+1}, ... , x_{-1}, x)$
    - 타겟 토큰이 발생하기 전까지의 Sequence 세기 : $p(x_{-N}, x_{-N+1}, ... , x_{-1})$
    - 예시 (Tri-gram) : *"New York"* 이 주어졌을 때 *"University"* 가 나올 확률 

    1. 데이터에 존재하는 *New York University* 의 개수를 모두 센다.
    2. *"New York (1 word) "* 이 등장하는 횟수를 모두 센다.
    3. 2에서 1의 결과로부터 최대 우도를 추정

$$
p(x \vert x_{-N}, x_{-N+1}, ... , x_{-1}) = \frac{p(x_{-N}, x_{-N+1}, ... , x_{-1}, x)}{\sum_{x \in V}p(x_{-N}, x_{-N+1}, ... , x_{-1}, x)} \\ \qquad \qquad \qquad \qquad \quad \quad \quad \quad \sim \frac{p(x_{-N}, x_{-N+1}, ... , x_{-1}, x)}{\sum_{x^\prime \in V}p(x_{-N}, x_{-N+1}, ... , x_{-1}, x^\prime)}
$$



- N-gram 모델에서 발생하는 문제점

  1. **Data sparsity** : Lack of Generalization. 아래 예시(Tri-gram)에서 나온 문장을 보자. 만약 데이터에 *"chasing a llama"* 라는 phrase가 없다면 해당 문장이 나올 Probability는 0이 된다.  

  $$
  \text{Ex) } p(\text{a lion is chasing a llama}) \\= p(\text{a}) \times p(\text{lion}\vert\text{a}) \times p(\text{is}\vert\text{a lion}) \times p(\text{chasing}\vert\text{lion is}) \times p(\text{a}\vert\text{is chasing}) \times p(\text{llama}\vert\text{chasing a}) = 0 \\ \because p(\text{llama}\vert\text{chasing a}) = 0
  $$

  

  - sparsity를 해결하기 위한 방법들

    - Smooting : 0을 피하기 위해서 아주 작은 수 $(\epsilon)$ 를 대입해 준다. 그러나 chasing a universe, chasing a star보다는 chasing a llama가 실제로 더 많이 나올거 같은데 모두 같은 취급을 한다면 둘을 구분하지 못한다는 또 다른 문제 발생하게 된다. 

    $$
    p(x \vert x_{-N}, x_{-N+1}, ... , x_{-1}) \sim \frac{c(x_{-N}, x_{-N+1}, ... , x_{-1}, x) + \epsilon}{\epsilon\vert V \vert + \sum_{x^\prime \in V}p(x_{-N}, x_{-N+1}, ... , x_{-1}, x^\prime)}
    $$

    - Backoff : $p = 0$ 이 되는 경우에만 더 작은 N을 쓰도록 하는 방법이다. 해당 경우에는 적당한 보정치를 위해서 Correction factor $(\alpha, \beta)$ 를 취해준다.

    $$
    c(x_{-N}, ... ,x) = \begin{cases}{\alpha}c(x_{-N+1}, ... ,x) + \beta, \quad \text{if } c(x_{-N+1}, ... ,x) = 0 \\ c(x_{-N}, ... ,x) \qquad \qquad \text{ otherwise}\end{cases}
    $$

    - 가장 널리 적용되는 방법은 Kneser-Ney smoothing/backoff 이다.
    - KenLM 패키지 사용을 추천

  

  2. Long-term dependency 를 찾아낼 수 없다. 멀리 떨어진 단어끼리의 의존성을 찾기 위해서는 N을 늘려야하는데 N을 무작정 키우다보면 1에서 제기되었던 Sparsity 문제가 점점 커진다. 따라서 N-gram 모델에서는 이 문제를 해결하기 어렵다.

  - Dependency parsing(의존성 분석)을 통해 해결한다. 의존성 분석이란 문장의 문법적 구조를 파악하여 각 단어별 관계성을 찾는 방법을 나타낸다.  
    - Transition-based Model : Shift-Reduced
    - Graph-based Model : Dependency structure를 graph로 표현



# 4) Neural N-gram Language Model

## Neural N-gram Language Model

$$
\left[{\begin{array}{ccc} x_{-N} \\ x_{-N+1} \\ ... \\ x_{t-1}  \end{array}} \right] \cdot W : \text{ Table Lookup } \Rightarrow \text{ Sentence Representation Extractor } \vert \text{ Concatenation} \\ \Rightarrow \text{ An arbitrary sub-graph }(\theta) \Rightarrow \text{Softmax Classifier} \\ \text{ }
$$

- Discrete space(이산 공간)에서 count를 하게 되면 Data sparsity가 발생

  - 이전 예시 ( *"a lion is chasing a llama"* ) 에서

  $$
  \text{문제되는 것은 } c( \text{chasing a llama} ) = 0 \\
  c( \text{chasing a cat}) \gg 0 \text{ , } c( \text{chasing a dog}) \gg 0 \text{ , } c( \text{chasing a deer}) \gg 0 \text{ 이므로}
  $$

  - *llama* 가 *cat, dog, deer* 와 비슷하다는 것만 알 수 있다면 *chasing a llama* 를 *chasing a cat, chasing a dog, chasing a deer* 와 비슷하게 취급할 수 있다. 



- Neural N-gram Language Model은 **Continuous Vector Space** (연속적인 벡터 공간)에서 Similarity를 학습한다. 비슷한 pharse가 들어오면 비슷한 아웃풋을 내놓는다.
  $$
  D(x_t \vert x_{t-N}, ...,x_{t-1} \vert \vert x_{t} \vert x^\prime_{t-N}, ... , x^\prime_{t-1}) < \epsilon
  $$

  - 예시
    - *"there are **three teams** left for qualification"*
    - *"there are **four teams** left for qualification"*
    - *"there are **four groups** left for qualification"*
    - *"**three groups**"* (bi-gram)
      - Discrete space 에서는  = 0
      - Continuous vector space 에서는 != 0



- Neural N-gram Language Model이 작동하는 방식

1. 말뭉치로부터 모든 N-gram들을 수집
2. 그것들을 섞어 학습용 데이터세트를 만듦
3. 미니배치 SGD를 사용하여 Neural N-gram Language Model을 학습
4. 검증 세트에 맞추어 Early-stop
5. Test set에 대하여 Perplexity를 기록 (Perplexity가 낮으면 낮을수록 좋은 모델)

$$
\text{ppl} = b^{\frac{1}{\vert D \vert} \sum_{(x_1, ... , x_N) \in D} \log_b p(x_N \vert x_1, ... , x_{N-1})}
$$



## Long-Term Dependency

- Discrete representation 에서는 Data sparsity 문제 때문에 N을 쉽게 늘리지 못했다. 그래서 Long-term dependency가 발생하였다. 하지만  Continuous representation 에서는 sparsity 문제를 해결했기 때문에 N을 좀 더 넉넉하게 가져감으로써. Long-term dependency 문제를 어느정도 해결할 수 있다. 
  (N이 늘어날수록 파라미터의 개수도 늘어나고 그럴수록 다시 더 많은 데이터가 필요하므로 무한정 늘릴 수는 없음)



- Context의 크기를 늘리기
  - Convolutional Language Models : 기존 CNN에서는 window size를 빠르게 확장하기 어렵다. N을 늘리기 위해서는 많은 층을 쌓아야 하는데 그러면 발생하는 파라미터들이 너무 많아진다. 이런 문제를 해결하기 위해서 새로운 기법 도입
    - Dilated Convolution : n칸씩 뛰어가면서 커버를 하기 때문에 빠르게 N을 증가시킬 수 있다. 층을 겹치면 가운데 비는 토큰의 정보도 얻어올 수 있다. Dilated 뿐만 아니라 모든 CNN의 문제는 미래의 토큰 $x_{t+N}$ 까지 정보를 얻는 문제가 발생한다는 것.
    - Causal Convolution : 미래의 토큰들은 사용하지 않도록 masking out을 해준다. (하나씩 전진하는 Generative story를 어기지 않아야 함)
    - 이러한 모델들의 예시로 : ByteNet (PixelCNN, WaveNet) , Gated Convolutional Language Model 등이 있다.



- Context의 크기를 무한대로 늘리기
  - CBoW Language Models
    - Neural Language Model에서 *'concat'* 과정을 *'average'* 로 바꾸면 된다.
    - 효율적이지만 순서를 무시하기 때문에 Language Model로서는 적절하지 못하다.
  - Recurrent Language Models
    - 이전의 모든 토큰을 벡터로 변환한 뒤 그것을 기반으로 새로 나올 토큰을 예측
    - 온라인 학습으로 진행되기 때문에 효율적이다. 시간복잡도가 낮고 메모리 사용량이 적다.
  - Recurrent Memory Networks
    - **Recurrent Memory** 는 전체 Context를 일정한 크기의 벡터로 압축한다.
    - **Self-Attention** 은 압축하지 않으면서도 Long-term Dependancy를 잡아낼 수 있다.
    - 둘을 조합한 것이 **Recurrent Memory Networks(RMN)** 



[^1]: Likelihood(우도, $L(\theta \vert x)$ ) : 확률 밀도 함수 $f(x \vert \theta)$ 를 랜덤변수의 값 $x$ 의 함수가 아닌 파라미터 $\theta$ 의 함수로 보는 것. 확률 분포로부터 특정한 샘플 값 $x$가 발생하였을 때, 이 샘플 값 x가 나오게 하는 파라미터 $\theta$ 의 가능성. $\theta$ 값을 미리 알고 있을 때에는 확률 밀도 함수를 사용하지만, $x$ 만을 알고 있을 때에는 우도를 사용한다. (출처 : [데이터 사이언스 스쿨](https://datascienceschool.net/view-notebook/79140e6a9e364bcbb04cb8e525b9dba4/) )

