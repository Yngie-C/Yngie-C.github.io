---
layout: post
title: 분산 표현 (Distributed Representation) 방법
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# Word Embedding

단어를 벡터화 하는 가장 단순하고 직관적인 방법은 **원-핫 인코딩(One-hot Encoding)** 이다. 원-핫 인코딩은 문서에서 전체에 있는 단어 개수 만큼의 크기를 갖고 모든 요소가 0인 벡터를 만든 뒤 단어가 등장하는 순서에 따라 해당 인덱스를 1로 변환하는 방식이다. 예를 들어 *"I go to school"* 에 있는 단어들을 원-핫 인코딩으로 나타내면 아래와 같다.


$$
\qquad \text{I} : \left[{\begin{array}{cccc} 1 & 0 & 0 & 0  \end{array}}\right] \\
\quad \text{go} : \left[{\begin{array}{cccc} 0 & 1 & 0 & 0  \end{array}}\right] \\
\quad \text{to} : \left[{\begin{array}{cccc} 0 & 0 & 1 & 0  \end{array}}\right] \\
\text{school} : \left[{\begin{array}{cccc} 0 & 0 & 0 & 1  \end{array}}\right]
$$

원-핫 인코딩은 단순하고 쉽다는 장점이 있다. 하지만 모든 단어 벡터끼리의 내적이 항상 0이 나오기 때문에 단어 간의 관계를 파악하기 어렵다는 단점을 가지고 있다. 이런 문제를 해결하기 위한 방법으로 **분산 표현(Distributed Representation)** 방법이 있다. 단어를 벡터로 분산 표현하는 방법, 즉 **임베딩(Embedding)** 은 단어를 문서 내 단어의 개수 $\vert V \vert$ 보다 훨씬 작은 숫자인 $n$ 차원의 벡터로 나타낸다. 또한 벡터 내 성분이 연속적인 숫자이기 때문에 단어 벡터의 내적 값이 0이 아니게 되고 덕분에 단어 간의 의미론(Semantic)적인 관계가 보존된다.

<p align="center"><img src="https://www.offconvex.org/assets/analogy-small.jpg" alt="embedding" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.offconvex.org/2015/12/12/word-embeddings-1/">offconvex.org</a></p>



## NNLM

**신경망 언어 모델(Neural Network Language Model, NNLM)** 은 가장 고전적인 분산 표현 방식이다. 각 단어를 분산 단어 피처 벡터로 표현한다. 이 피처 벡터끼리의 결합 확률분포 함수(Joint probability function)을 이용하여 단어 시퀀스가 발생할 확률을 계산할 수 있다. 어떤 단어 피처 벡터가 좋은지와 이를 통해서 나타나는 확률 함수의 파라미터를 동시에 학습할 수 있다.

NNLM에서는 *(dog, cat), (the, a), (bedroom, room), (is, was), (running, walking)* 의 단어쌍이 의미적, 구조적으로 유사한 역할을 하고 있다면 우리는 이를 사용해 *The cat is walking in the bedroom* 이라는 문장을 다음과 같이 일반화 해볼 수 있다.

*A dog was running in a room, The cat is running is a room, A dog is walking in a bedroom, The dog was walking in the room ...*

NNLM이 Count 기반의 언어 모델과 다른 점은 무엇인가? Count 기반의 언어 모델에서 단어 시퀀스는 Chain rule에 의해 아래와 같이 Factorization 될 수 있고, 특히 N-gram 모델에서는 마르코프 가정(Markov Assumption)을 사용하여 타깃 단어 이전에 있는 몇 단어에 대해서만 계산한다.



$$
p(w_1, \cdots , w_T) = \prod^T_{i=1} p(w_t|w_1, \cdots , w_{t-1}) \\
p(w_t|w_1, \cdots , w_t) \approx p(w_t|w_{t-n}, \cdots, w_{t-1})
$$



NNLM은 위 방식과는 다르다. NNLM 에서는 단어를 $n(<< \vert V \vert)$ 차원 벡터 공간 내의 밀집 벡터(dense vectors)로 표현한다. 그리고 신경망이 특정 단어 다음에 올 단어 벡터가 어떤 것이 될 지에 대해서 훈련하게 된다.


$$
p(w_t = j|w_1, \cdots , w_{t-1}) = \frac{\exp (p^j \cdot g(x_1, \cdots, x_{t-1} + q^j))}{\sum_{j^\prime \in V} \exp (p^{j^\prime} \cdot g(x_1, \cdots, x_{t-1} + q^{j^\prime}))} \\ \qquad \quad \qquad = \text{softmax} (P_{g(x_1, \cdots, x_{t-1})} + q)
$$


NNLM의 구조는 아래 이미지와 같다. 각 단어의 인덱스로 미리 준비된 행렬 $C$ 를 Table Look-up 하여 행렬곱(Matmul)을 취해준다. 행렬곱의 결과에 softmax 함수를 취해주어 각 단어가 해당 자리에 위치할 확률을 구해낸다. 

<p align="center"><img src="http://i.imgur.com/vN66N2D.png" alt="nnlm" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

NNLM을 학습하는 주 목적은 좋은 모델 함수 $f(w_t, \cdots, w_t-n+1)$ = \hat{P} (w_t \vert w_1^{t-1}) 을 찾는 것이다. 주어진 단어 시퀀스 뒤에 오는 $t$ 번째 단어의 확률이 극대화 되는 $f$ 를 찾는 것이 목적이다. NNLM을 적용하기 위해서 두 가지 제약 조건이 있다. 첫 번째로 t-1 자리에 어떤 단어 $w_{1}^{t-1}$ 가 나오더라도 t에 나오는 모든 단어의 확률 합은 1이 되어야 한다. 그리고 각 단어가 생성될 확률은 모두 0보다 크거나 같아야 한다. 이 두 가지 제약 조건을 식으로 나타내면 다음과 같다.


$$
\sum^{\vert V \vert} _{i=1} f(i, w_{t-1}, \cdots, w_{t-n+1}) = 1 \\
f \geq 0
$$


모델링된 함수 $f$ 를 아래와 같이 수식을 Table look-up 행렬 $C_{\vert V \vert \times n}$ 와의 Matmul 을 거쳐 변환된 벡터로 표현되는 새로운 함수 $g$ 로 나타낼 수 있다. 여기서 C는 $n (<< \vert V \vert)$ 차원의 벡터이다. 말이 되는(가능성이 높은) 시퀀스에게는 높은 확률을 부여하고 말이 안되는(가능성이 낮은) 시퀀스에게는 낮은 확률을 부여하는 것이 확률 함수 g의 역할이다. 


$$
f(i, w_{t-1}, \cdots, w_{t-n+1}) = g(i, C(w_{t-1}), \cdots , C(w_{t-n+1}))
$$


아래 수식과 같이 훈련 말뭉치의 로그-우도(log-likelihood)를 최대화 하는 과정으로 훈련이 이루어진다.


$$
L = \frac{1}{T} \sum_t \log f(i, w_{t-1}, \cdots, w_{t-n+1};\theta) + R(\theta)
$$

은닉층이 내놓는 값 $y$ 는 아래와 같이 표현할 수 있다. 


$$
y = b + Wx + U \cdot \tanh(d+Hx) \\
b : \text{출력값의 편향,} \quad d : \text{은닉층의 편향} \\
U : \text{은닉층-출력층의 가중치,} \quad W : \text{단어 피처-출력값의 가중치,} \quad H : \text{은닉층의 가중치}
$$


각 단어의 확률이 가장 높을 때의 모델을 구현하는 것이 목적이므로 확률적 경사 상승법(Stochastic Gradient Ascent, SGA) 를 사용하여 파라미터 셋 $\theta$ 를 개선해나간다.


$$
\theta \leftarrow \theta + \epsilon \frac{\partial \log \hat{P} (w_t|w_{t-1}, \cdots, w_{t-n+1})}{\partial \theta}
$$


## Word2Vec

**Word2Vec** 에는 **CBoW(Continuous Bag-of-Words)** 와 **Skip-gram** 의 두 가지 방법이 있다. 전자는 주변 단어(Context, 문맥)로부터 타깃 단어 $w_t$ 를 예측하고 후자는 타깃 단어로부터 주변 단어를 예측한다. 얼핏 생각해보면 더 많은 정보를 받아 하나의 단어를 예측하는 CBoW의 성능이 더 좋을 것으로 예측하기 쉽다. 하지만 실제로는 역전파가 일어나는 관점에서 생각해보면 CBoW는 하나의 단어에서 오는 정보를 여러 단어에 흘려주고, Skip-gram은 여러 단어로부터 흘러나오는 역전파 정보를 한 단어가 흘려받는다. 때문에 Skip-gram의 성능이 보통 더 잘 나오는 편이다. 아래는 CBoW와 Skip-gram모델을 그림으로 도식화한 것이다.

(이미지)

Skip-gram이 진행되는 과정을 살펴보자. 내부에 활성화 함수가 없는 선형적(Linear)인 구조를 가지고 있다. Skip-gram의 목표는 타깃 단어 $w_t$ 로부터 주변 단어를 예측하는 모델을 만드는 것이므로 아래와 같이 목적 함수를 표현할 수 있다.


$$
J(\theta) = \frac{1}{T} \sum^T_{t=1} \sum_{-m \leq j \leq m, j \neq 0} \log p(w_{t+j}|w_t)
$$


위 목적 함수에서 $m$ 은 Window의 크기를 나타내며 주변 단어의 확률을 최대화 하는 파라미터 셋 $\theta$ 를 구하는 것을 목적으로 한다. 각 주변 단어의 확률은 다음과 같이 구해진다. 한번에 역전파 정보를 받는 것과 따로 받아 더해 준 결과가 같으므로 실제 역전파가 흘려받을 때는 각 주변 단어에서 하나씩 흘려받은 결과를 더해준다.

예를 들어, *"The quick brown fox jumps over..."* 로 시작하는 문장을 크기가 좌우 2인 윈도우로 학습한다면 *(the, samples), (the, brown)* / *(quick, the), (quick, brown), (quick, fox)* / *(brown, the), (brown, quick), (brown, fox), (brown, jumps)* 처럼 따로 단어 쌍을 만든 후 학습이 진행된다.


$$
p(c|w) = \frac{\exp(x_w^Tv_c)}{\sum^K_{k=1} \exp(x_w^Tv_k)} \\
x_w : \text{단어 피처 벡터} \quad v_c : \text{단어 분류기 벡터}
$$


CBoW는 주변 단어로부터 정보를 받아 타깃 단어를 예측하므로 아래와 같이 단어의 확률을 나타낼 수 있다. 여러 단어의 정보를 받기 때문에 대문자 $C$ 로 나타낸다.


$$
p(w|C) = \frac{\exp(h_c^Tv_w)}{\sum^K_{k=1} \exp(h_c^Tv_k)} \\
h_C : \text{주변 단어 피처 벡터} \quad v_w : \text{단어 분류기 벡터}
$$


다시 Skip-gram으로 돌아와 구조를 더 자세히 들여다 보자. 수식의 단순화를 위하여 $w_{t+j} \rightarrow o$ , $w_t \rightarrow c$ (outside, center)로 나타내면 Skip-gram의 확률 식을 아래와 같이 나타낼 수 있다.


$$
p(o|c) = \frac{\exp(u_o^Tv_c)}{\sum^W_{w=1} \exp(u_w^Tv_c)}
$$


이 확률을 최대화 하는 벡터인 $v_c$ 를 찾는 것이 목적이므로 위 수식에 로그를 취한 것을 $v_c$ 로 미분한 수식을 살펴보도록 하자.


$$
\frac{\partial}{\partial v_c} \log p(o|c) = \frac{\partial}{\partial v_c} \log \frac{\exp(u_o^Tv_c)}{\sum^W_{w=1} \exp(u_w^Tv_c)} \qquad \qquad \\
\qquad \qquad \quad = \frac{\partial}{\partial v_c} u_o^Tv_c-\frac{\partial}{\partial v_c} \log \sum^W_{w=1} \exp(u^T_w v_c) \\
\qquad \qquad \qquad \qquad \quad = u_o - \frac{1}{\sum^W_{w=1} \exp(u^T_w v_c)} \cdot \sum^W_{w=1} \exp(u^T_w v_c) \cdot u_w \\
= u_o - \sum^W_{w=1} P(w|c) \cdot u_w \\
\because - \frac{1}{\sum^W_{w=1} \exp(u^T_w v_c)} \cdot \sum^W_{w=1} \exp(u^T_w v_c) = - \sum^W_{w=1} \frac{\exp(u^T_w v_c)}{\sum^W_{w=1} \exp(u^T_w v_c)} = - \sum^W_{w=1} P(w|c)
$$


확률을 최대화 하는 것이 목적이므로 경사 상승법을 적용하여 가중치 벡터 $v_c$ 를 개선해나간다.


$$
v_c(t+1) = v_c(t) + \alpha(u_o - \sum^W_{w=1} P(w|c) \cdot u_w)
$$


가중치의 개수는 $2 \times V \times N$ 으로 매우 크다. 가중치의 개수가 많으면 학습 시간이 오래걸리고 컴퓨팅 자원도 많이 사용되므로 가중치의 개수를 줄이기 위해 여러 전략을 취한다. 먼저 자주 등장하는 구(phrase)는 하나의 단어로 취급한다. 두 번째로, 너무 자주 등장하는 단어는 학습 데이터에서 누락시킨다. 고빈도 단어를 누락시키는 확률 $P$ 에 대한 식은 다음과 같다.


$$
P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} \\
f(w_i) : w_i \text{가 등장하는 빈도,} \quad t : \text{threshold}
$$


**네거티브 샘플링(Negative Sampling)** 도 계산량을 줄이기 휘한 방법이다. Skip-gram은 한 단어를 구하기 위해서 윈도우 내 모든 단어에 대해 전부 계산을 한다. 모든 단어의 소프트맥스(Softmax)를 계산할 경우 계산량이 많아지므로 그 중 일부를 덜어낸다. 계산 대상이 단어 전체 $\vert V \vert$ 에서 이 때 목적 함수의 수식을 아래와 같이 나타낼 수 있다.


$$
J(\theta) = \frac{1}{T} \sum^T_{t=1} \sum_{-m \leq j \leq m, j \neq 0} \log p(w_{t+j}|w_t) = \frac{1}{T} \sum^T_{t=1} J_t (\theta) \\
J_t (\theta) = \log \sigma(u_o^T v_c) + \sum_{i=1}^T E_{i \sim P(W)} [\log \sigma(-u_i^T v_c)] \\
P(w_i) = \frac{f(w_i)^{3/4}}{\sum_{j=0}^n (f(w_i)^{3/4})}
$$


## GloVe

Word2Vec의 한계는 자주 사용되는 단어에 대해서 너무 많이 계산한다는 데 있다. *'the'* 와 같이 빈도가 매우 높은 단어가 있을 경우 학습에 불균형이 발생하기 마련이다. 몇 가지 샘플링 방식을 통해 계산량을 보정해 줄 수는 있지만 계산 문제를 완벽하게 해결 해주지는 못한다. 이를 원천적으로 해결하기 위해 등장한 것이 **GloVe** 이다. 아래 예시를 통해 GloVe 를 알아보자.

| Prob. and Ratio.                                        | k = solid            | k = gas              | k = water            | k = fashion          |
| ------------------------------------------------------- | :------------------- | -------------------- | -------------------- | -------------------- |
| $P(k \vert \text{ice})$                                 | $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
| $P(k \vert \text{steam})$                               | $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
| $\frac{P(k \vert \text{ice})}{P(k \vert \text{steam})}$ | 8.9                  | $8.5 \times 10^{-2}$ | 1.36                 | 0.96                 |

위 표는 *ice* 와 *steam* 이 각각 등장했을 때 *solid, gas, water, fashion* 이 등장하는 빈도를 나타낸 표이다. 표의 3번째 행에서 볼 수 있듯 *solid* 가 등장하는 빈도는 글에 *steam* 보다 *ice* 가 등장했을 때 훨씬 높다. 반대로 *gas* 가 등장하는 빈도는 *ice* 보다 *steam* 이 등장했을 때 더 낮다. *water, fashion* 이 등장하는 빈도는 글에 *ice* 가 나오든 *steam* 이 나오든 큰 상관이 없다.

위 3번째 행을 일반화한 $\frac{P(k|i)}{P(k|j)}$ 을 하나의 함수 $F$ 로 나타내보자.


$$
F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}} = \frac{P(k|i)}{P(k|j)}
$$


이 함수를 적당히 변형시켜 각각의 단어 벡터를 빼준 값과 단어 k의 벡터를 내적한 것을 변수로 한다고 하면 다음과 같이 나타낼 수 있다.


$$
F((w_i - w_j)^T \tilde{w}_k) = \frac{P_{ik}}{P_{jk}} = \frac{P(k|i)}{P(k|j)}
$$



이 $F$ 를 이용하여 표 3행에 있는 $\frac{P(k \vert \text{ice})}{P(k \vert \text{steam})}$ 와 역수를 나타내면 아래와 같이 나타낼 수 있다.


$$
\frac{P(\text{solid} \vert \text{ice})}{P(\text{solid} \vert \text{steam})} = F((\text{ice} - \text{steam})^T\text{solid}) \\
\frac{P(\text{solid} \vert \text{steam})}{P(\text{solid} \vert \text{ice})} = F((\text{steam} - \text{ice})^T\text{solid})
$$


**준동형 사상(Homomorphism)** 이라는 수학적 방법을 사용하면 함수 $F$ 의 구조를 유추해볼 수 있다. 아래의 수식을 살펴보면, 위 두 식에서 함수 내 변수는 덧셈에 대한 역원 관계에 있고, 함숫값은 곱셈에 대한 역원으로 표현됨을 알 수 있다. 


$$
(\text{ice} - \text{steam})^T\text{solid} = - (\text{steam} - \text{ice})^T\text{solid} \\
F((\text{ice} - \text{steam})^T\text{solid}) = \frac{1}{F((\text{steam} - \text{ice})^T\text{solid})}
$$

결국 $F$ 는 $F(a+b) = F(a)F(b)$ 를 만족하는 함수여야 하므로 $e^x$ 형태의 지수함수가 된다. $F(x) = \exp(x)$ 로 두고 단어 벡터의 내적 값을 아래와 같이 $\log X_{ik}$ 에 대한 값으로 나타내 줄 수 있다.


$$
w_i^T\tilde{w}_k = \log P_{ik} = \log X_{ik} - \log X_i \\
w_i^T\tilde{w}_k = \log X_{ik} - b_i - \tilde{b_k} \\
w_i^T\tilde{w}_k + b_i + \tilde{b_k} = \log X_{ik}
$$


위 식을 이용하여 Glove의 목적 함수를 쓸 수 있다.


$$
J = \sum_{i,j=1}^V (w_i^T\tilde{w}_j + b_i + \tilde{b_j} - \log X_{ij})^2 \\
\Rightarrow J = \sum_{i,j=1}^V f(X_{ij})(w_i^T\tilde{w}_j + b_i + \tilde{b_j} - \log X_{ij})^2
$$


여기에서 추가되는 함수 $f$ 는 3가지 조건을 만족해야 한다. $f(0) = 0$ 이어야 하고 동시 발생 확률이 낮은 단어를 감소시키면 안된다. 마지막으로 높은 $x$ 에 대해서는 ~ 해야한다. 이러한 함수 $f$ 를 사용하여 고빈도 단어의 영향력을 줄여준다. 



## Fasttext

NNLM, Word2Vec, Glove 는 모두 각 단어가 가진 형태적(morhology) 특성을 무시하고 있다는 한계를 가지고 있다. 때문에 터키어, 핀란드어처럼 형태적으로 복잡한 언어에는 잘 작동하지 않는다. **Fasttext** 는 이러한 단점을 해결하기 위해서 character 레벨에서부터 접근한다. character N-gram을 학습하며 단어는 character N-gram 벡터들의 합으로 표현한다.

Fasttext에서 사용되는 서브워드 모델은 아래와 같이 표현할 수 있다. 아래 수식에서 $G_w \subset {1, \cdots , G}$ 이다.


$$
\text{score}(w,c) = \sum_{g \in G_w} \mathbf{z}_g^T\mathbf{v}_c
$$


이런 방법으로 단어를 character N-gram 벡터의 값으로 표현한다. N-gram의 사이즈는 3,4,5,6 이며 단어마다 같은 character 시퀀스를 가지고 있더라도 평균을 내기 때문에 서로 다른 값으로 표현된다.



## Sentence/Paragraph/Document-level

추후에 추가하는 것으로