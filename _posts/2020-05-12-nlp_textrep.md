---
layout: post
title: Count 기반의 표현 (Count based Representation)
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# Count Based Representation

우리가 처리하고자 하는 문서는 각각의 길이가 모두 다르다. 이렇게 가변 길이로 이루어진 문서를 컴퓨터에 집어넣기 위해서는 고정 길이의 벡터로 변환해주어야 한다. 머신러닝에서는 모두 길이가 동일한 인스턴스를 다루기 때문이다. 문서를 벡터로 표현하는 방법은 크게 Count based 와 Distributed  두 가지로 나눠볼 수 있다. 여기서는 수(Count)를 기반으로 문서를 벡터로 변환하는 방법에 대해 알아고보자 한다. Bag of Words, Word Weighting, N-gram에 대해 알아보자.



## Bag of Words

**백오브워즈(Bag of Words)** 는 단어를 문서로 표현하는 방법 중 가장 고전적인 방식이다. 문서는 단어의 집합체이므로 원하는 문서에 단어가 있는지 없는지 또는 몇 번이나 등장하는지를 세어볼 수 있다. 이렇게 헤아린 단어의 빈도로 뭔가를 추론할 수 있을 것이라는 전제로부터 출발한다. 분석하고자 하는 문서(Document)를 각 열에 쓰고 각 행에는 각각의 단어(Term)을 쓴 뒤 갯수를 세아려 기록한 행렬을 **Term-Document Matrix(TDM)** 라고 한다. 이를 통해서 각 문서에 특정 단어가 등장하는지/아닌지 혹은 몇 번 등장하는지를 알 수 있다. (TDM을 전치시킨 Document-Term Matrix를 사용할 수도 있다)

<img src="https://user-images.githubusercontent.com/45377884/81645203-206ee380-9464-11ea-81ba-8a8fb9e52f03.png" alt="tdm" style="zoom:80%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

위 그림에 있는 두 가지 TDM 중 Binary representation은 문서에 단어가 등장하는지 아닌지를, Frequency representation은 문서에 단어가 몇 번이나 등장하는 지를 기록한다. Frequency representation이 더 많은 정보를 제공해주지만 항상 Frequency representation을 사용하지는 않는다. 특정 모델에 대해서는 Binary representation이 더 좋은 결과를 내어주기도 하기 때문에 문제의 성질에 맞는 표현 방식을 사용하여야 한다.

Bag of Words의 단점은 단어의 순서를 무시한다는 데에 있다. 순서를 무시하기 때문에 *"John loves Mary"* 와 *"Mary loves John"* 을 표현하면 같은 표현이 된다. 반대로 특정 벡터 표현이 주어졌을 때 이게 *"John loves Mary"* 가 변환된 것인지 *"Mary loves John"* 에서 변환된 것인지 아는 것이 불가능하다.



## Word Weighting

**단어 가중치 부여(Word Weighting)** 는 특정 문서에서 더 중요한 역할을 하는 단어에 가중치 부여하는 방법이다. TF와 DF라는 두 수치의 결합을 기본으로 한다. 먼저 **TF(Term Frequency)** 는 우리가 분석하고자 하는 문서에 등장하는 단어의 숫자를 나타낸 수치이다. 빈도가 높을수록 그 단어가 더 중요할 것이라는 가정을 이어간다. **DF(Document Frequency)** 는 특정 단어가 코퍼스 내에 있는 전체 문서 중 몇 개의 문서에 등장했는지를 나타낸다. DF가 낮을수록, 즉 등장하는 문서의 수가 적을 수록 더 중요한 단어일 확률이 높다는 가정으로부터 출발한다. 이를테면, *is, can, the* 등의 단어는 어디에든 많이 등장하기 때문에 TF가 매우 높아 TF만 사용하면 중요도가 높은 단어로 취급된다. 하지만 모든 문서에 등장하기 때문에 DF가 높아 중요도가 낮은 단어로 보정할 수 있다. 실제 계산에서는 DF에 역수를 취하고 로그를 취한 IDF(Inverse Document Frequency)를 사용한다.  

결국은 토큰이 특정 문서에 얼마나 중요한지를 나타낸다는 것은 그 단어의 TF는 크고 DF는 작다(IDF는 크다)는 것이다. 실제로는 이 두 값을 결합한 **TF-IDF** 를 사용한다.



$$
\text{TF-IDF(w)} = \text{TF(w)} \times \log \frac{N}{\text{DF(w)}}
$$



토큰을 특성으로 하는 인스턴스로 각 문서를 $\vert V \vert$ (총 단어 수) 차원을 가진 벡터 스페이스로 표현할 수 있다. 단어의 개수가 많아지면 벡터의 차원이 높아지고 Sparsity 문제가 발생한다.

TF-IDF에도 각각 여러가지 변종들이 존재한다. 일반적으로는 TF에서 l(logarithm)를, IDF에서는 t(idf)를 사용한다. 하지만 해결해야 하는 문제에 따라 다양하게 바꾸어 가며 가장 좋은 조합을 찾아 나가야 한다.

<img src="https://user-images.githubusercontent.com/45377884/81643718-1b5c6500-9461-11ea-8e30-c41cbc1e6dc7.png" alt="tf-idf" style="zoom:80%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

## N-gram

Count 기반의 문서 표현 방법의 마지막은 **엔그램(N-gram)** 에 기반한 방법이다. N-gram이란 확률을 추정할 때 전체 단어를 조합하는 대신 일부 단어 조합의 출현 빈도만을 계산하는 방법이다. 전체 텍스트를 조합하지 않는 이유는 희소성(Sparsity) 문제가 발생하기 때문이다. N이 7~8정도만 넘어가더라도 말뭉치에 단어 조합이 없는 경우가 많다. 이 때 대부분의 단어 조합 확률은 0으로 나오게 된다. 이런 문제를 해결하기 위해서 조합하는 단어의 수를 줄여야만 하는데 이 때 얼마만큼의 단어 조합을 볼 것인지를 N-gram으로 정하게 된다. N의 크기에 따라 Uni(1)-gram, Bi(2)-gram, Tri(3)-gram, 4-gram, 5-gram... 등이 있다.

이렇게 할 수 있는 이유는 **마르코프 가정(Markov Assumption)** 때문이다. 마르코프 가정이란 특정 시점에서 어떤 상태의 확률은 그 직전 상태에만 의존한다는 논리이다. 마르코프 가정을 하게 되면 이전에 있는 모든 시퀀스를 탐색하지 않고도 타겟 단어의 확률을 구해낼 수 있다. 수식으로 나타내면 아래와 같다.



$$
P(x_i|x_1, x_2, \cdots , x_{i-1}) \approx P(x_i|x_{i-k}, \cdots , x_{i-1})
$$



N-gram 에서 사용하는 $N = k+1$ 이므로 Bi(2)-gram, Tri(3)-gram의 식은 다음과 같이 나타낼 수 있다.



$$
\text{Bi-gram : } P(x_i|x_1, x_2, \cdots , x_{i-1}) \approx P(x_i|x_{i-1}) \qquad \\
\text{Tri-gram : } P(x_i|x_1, x_2, \cdots , x_{i-1}) \approx P(x_i|x_{i-2}, x_{i-1})
$$



연쇄 법칙을 통해 N-gram이 단어의 확률을 구하는 방식을 확장하면 다음과 같이 한 문장의 확률도 구해낼 수 있다.



$$
P(x_1, x_2, \cdots , x_{i-1}, x_i) \approx \prod^n_{i=1} P(x_i|x_{i-k}, \cdots , x_{i-1})
$$



N-gram 방법은 출현하지 않는 단어 시퀀스를 어떻게 표현할까에 대한 문제를 가진다. 말뭉치에 출현하지 않는 단어 조합은 모두 확률이 0이 되기 때문이다. 물론 *"컵을 먹다", "책꽂이를 먹다"* 처럼 실제로 쓰지 않는 말도 있겠지만 *"마라룽샤를 먹다"* 처럼 말은 되지만 말뭉치가 충분히 크지 않아 등장하지 않는 단어들도 있는데 두 단어 시퀀스 모두 N-gram 모델에서는 0이 된다. 이런 문제를 해결하기 위해서 출현 빈도값을 보정해주는 것을 **스무딩(Smoothing)** 혹은 **디스카운팅(Discounting)** 이라고 한다. 기본적인 스무딩의 방법은 모든 단어 시퀀스의 출현 빈도에 작은 값을 더해주는 것이다. 수식으로 나타내면 다음과 같다.


$$
P(w_i|w_{<i}) \approx \frac{\text{Count}(w_{<i},w_i) + k}{\text{Count}(w_{<i}) + kV} \approx \frac{\text{Count}(w_{<i},w_i) + (m/V)}{\text{Count}(w_{<i}) + m}
$$


N-gram 모델의 성능을 높이기 위한 방법으로 **Kneser-Ney(KN) 디스카운팅** 이 있다. KN 디스카운팅의 기본적인 아이디어는 목적 단어 $w$ 이전에 나타나는 단어 $v$ 가 얼마나 다양한지를 알아내는 데에 있다. 다양한 단어 뒤에 나타나는 단어일수록 말뭉치에서 보지 못한 단어 시퀀스로 나타날 가능성이 높다는 내용이다. KN 디스카운팅에 있는 Score를 다음과 같이 모델링한다.


$$
\text{Score}_{\text{continuation}}(w) \propto |\{v:\text{Count}(v,w) > 0\}| \\
\text{Score}_{\text{continuation}}(w) = \frac{|\{v:\text{Count}(v,w) > 0\}|}{\sum_{w^{\prime}}|\{v:\text{Count}(v,w^{\prime}) > 0\}|}
$$


식에서 $w^{\prime}$ 은 $v$ 뒤에 등장하는 다른 단어들을 나타낸다. KN 디스카운팅을 사용하여 단어가 등장할 확률 $P_{KN}$ 을 모델링하면 아래와 같이 나타낼 수 있다. 아래 식에서 $d$ 는 상수이다. (보통은 0.75를 사용한다.)


$$
P_{KN} (w_i|w_{i-1}) = \frac{\max(\text{Count}(w_{i-1},w_i) - d,0)}{\text{Count}(w_i)} + \lambda(w_{i-1}) \times \text{Score}_{\text{continuation}}(w_i) \\
\text{where} \quad \lambda(w_{i-1}) = \frac{d}{\sum_v \text{Count}(w_{i-1})} \times |\{w:c(w_{i-1},v) > 0\}|
$$


가장 많이 사용되는 방법은 위 수식에 적당한 수정을 더해준 Modified-KN 디스카운팅이 많이 사용된다. 식은 복잡하지만 SRILM 이라는 툴킷에 구현되어 있기 때문에 그것을 사용하면 된다.

N-gram에서 단어 시퀀스가 길어질수록 실제 말뭉치에 등장하는 횟수가 급격히 줄어들게 된다. 백오프 방식은 이 문제를 해결하기 위해 도입한 방법이다. N이 클 때 작은 N의 조합으로 인터폴레이션(Interpolation, 보간) 함으로써 더 높은 일반화 효과를 얻을 수 있다.


$$
\tilde{P}(w_n|w_{n-k}, \cdots , w_{n-1}) = \lambda_1 P(w_n|w_{n-k}, \cdots , w_{n-1}) \qquad \qquad \\
\qquad \qquad \qquad \quad + \lambda_2 P(w_n|w_{n-k+1}, \cdots , w_{n-1}) \\
+ \cdots \qquad \quad \\
+ \lambda_k P(w_n), \\
\text{where} \quad \sum_i \lambda_i = 1
$$
