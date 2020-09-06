---
layout: post
title: 유전 알고리즘(Genetic Algorithm)
category: Machine Learning
tag: Machine-Learning
---



# Genetic Algorithm

**유전 알고리즘(Genetic algorithm)**은 메타 휴리스틱 방법론 중 하나의 방법입니다. 유전 알고리즘에 대해 알아보기 전에 먼저 메타 휴리스틱 방법론이 무엇인지에 대해서 알아보겠습니다.

### Meta Heuristics

먼저 휴리스틱에 대해 알아보겠습니다. **휴리스틱(Heuristics)**이란 무엇일까요? 위키피디아에서 말하는 휴리스틱의 뜻은 다음과 같습니다.

> *"**휴리스틱(heuristics)** 또는 **발견법**이란 불충분한 시간이나 정보로 인하여 합리적인 판단을 할 수 없거나, 체계적이면서 합리적인 판단이 굳이 필요하지 않은 상황에서 사람들이 빠르게 사용할 수 있게 보다 용이하게 구성된 간편추론의 방법이다."*

위에도 나오는 것처럼 휴리스틱은 간편 추론의 방법입니다. 그렇다면 메타 휴리스틱(Meta heuristics)은 무엇일까요? 위키피디아에서는 메타 휴리스틱을 다음과 같이 설명하고 있습니다.

> *특정문제가 갖는 정보에 크게 구속되지 않고 다양한 문제에 적용가능한 상위수준의 발견적 기법*

복잡한 문제를 풀기 위해서 시행착오(Trial and error) 방법을 사용할 때 효과적으로 문제를 풀기 위해서 사용합니다. 메타 휴리스틱을 적용한 최적화 알고리즘 중에는 자연계의 행동을 모사한 방법이 많습니다. 대표적인 경우가 뇌의 활동을 모사한 인공신경망(Artificial neural networks, ANNs), 개미의 이동을 모사한 개미 식민지 최적화(Ant colony optimization, ACO), 조류나 어류 무리의 행동을 모사한 입자 군집 최적화(Particle swarm optimization, PSO) 등이 있습니다.

 ### Genetic Algorithm

유전 알고리즘은 생물의 번식을 모사한 진화 알고리즘입니다. 번식 과정을 반복하면서 더 나은 솔루션을 찾아가며 기존의 솔루션을 보존합니다.

유전 알고리즘을 이전의 다른 방법들과 비교하여 시간-성능 그래프에 나타내면 다음과 같습니다. Stepwise selection 보다 더 오랜 시간이 걸리게 되지만 더 좋은 성능을 보이는 것을 알 수 있습니다.

<img src="https://user-images.githubusercontent.com/45377884/92316358-b17a5380-f02d-11ea-9ea3-6fa7a0ae62a9.png" alt="ga" style="zoom:50%;" />

유전 알고리즘은 다음과 같은 6단계로 진행됩니다. 아래에서 2-5단계에 해당하는 과정은 선택한 변수가 특정한 조건을 만족할 때까지 반복하는 과정입니다.

1. 크로모좀 초기화 및 파라미터 설정**(Initiation)**
2. 각 크로모좀 선택 변수별 모델 학습
3. 각 염색체 적합도 평가**(Fitness evaluation)**
4. 우수 염색체 선택**(Selection)**
5. 다음 세대 염색체 생성**(Crossover & Mutation)**
6. (조건을 만족하면) 최종 변수 집합 선택

그림으로 나타내면 아래와 같습니다.

<img src="https://static.packt-cdn.com/products/9781788472173/graphics/3fe7ae9e-a589-40a3-9904-c5cf0f650d94.png" alt="ga_img" style="zoom: 50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781788472173/8/ch08lvl1sec60/genetic-algorithms-structure">subscription.packtpub.com</a></p>

## Initialization

첫 번째 단계인 **Initialization(초기화)**에 대해서 알아보겠습니다. 유전 알고리즘은 생물의 유전 현상을 모사한 것이므로 사용하는 용어 역시 동일한 단어를 사용합니다. 그렇기 때문에 특성을 선택하는 경우 하나하나를 **크로모좀(Chromosome, 염색체)**라고 합니다. 데이터셋이 $d$ 차원 이라면 크로모좀 역시 $d$ 차원의 벡터가 됩니다. 각 특성을 선택하는 지의 여부를 $0,1$ 로 나타내는 바이너리 인코딩(Binary Encoding)을 사용합니다. 각각의 특성, 그리고 이와 연결된 인코딩 값은 유전자(Gene)라고 표기합니다.

Initialization 단계에서는 이후 알고리즘에서 사용할 도구를 하이퍼파라미터를 설정하여 정하게 됩니다. 첫 번째 하이퍼파라미터는 몇 개의 크로모좀으로 시작할 지를 결정하는 **Population** 입니다. 이 값을 100으로 설정하면 한 세대에서 100개의 크로모좀이 생성되므로 100개의 변수 부분집합을 평가할 수 있습니다.

두 번째로는 목적 함수에 해당하는 **피트니스 함수(Fitness function)**를 설정할 수 있습니다. 세 번째로는 교배 방식(Crossover mechanism)을 설정합니다. 네 번째로는 돌연변이 비율을 설정합니다. 마지막으로 종료 기준을 설정하면 모든 설정이 완료됩니다. 일반적으로 종료 기준은 피트니스 함수값의 유의미한 상승을 결정하는 임계값 또는 GA Operation의 최대 반복 횟수로 설정합니다.

각 크로모좀은 이진 벡터 형태이지만 이를 만드는 과정인 Population initialization은 각 값에 랜덤 값을 부여하는 것에서 시작합니다. $[0,1]$ 범위 내의 임의의 값을 각각의 Gene에 부여한 뒤에 Cut-off 값을 기준으로 이진 벡터로 변환합니다. 이런 과정을 통해 임의성을 부여할 수 있습니다. 아래는 Population이 8이고 특성 차원이 10인 경우에 Population initialization을 수행한 예시입니다.

<img src="https://user-images.githubusercontent.com/45377884/92330290-bc25fe80-f0a8-11ea-9a89-949c23f2b967.png" alt="population_init" style="zoom: 67%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/Business-Analytics-IME654-">github.com/pilsung-kang/Business-Analytics-IME654</a></p>

## Training

그 다음으로는 이렇게 설정된 크로모좀에 따라 각 모델을 훈련시킵니다. 위 사진의 예시를 따르면 다음과 같은 모델들을 학습하게 됩니다.


$$
\text{Model 1} : \hat{y} = \hat{\beta_0} + \hat{\beta_2}x_2 + \hat{\beta_3}x_3 + \hat{\beta_4}x_4 + \hat{\beta_6}x_6 + \hat{\beta_9}x_9 + \hat{\beta_{10}}x_{10} \\
\vdots \\
\text{Model 4} : \hat{y} = \hat{\beta_0} + \hat{\beta_3}x_3 + \hat{\beta_4}x_4 + \hat{\beta_5}x_5  + \hat{\beta_6}x_6 + \hat{\beta_8}x_8  + \hat{\beta_9}x_9 \\
\vdots \\
\text{Model 8} : \hat{y} = \hat{\beta_0} + \hat{\beta_2}x_2 + \hat{\beta_5}x_5 + \hat{\beta_7}x_7 + \hat{\beta_9}x_9 + \hat{\beta_{10}}x_{10}
$$

## Fitness Evaluation

어떤 크로모좀에 의해 만들어진 모델이 더 나은 것인가를 판단하기 위한 과정입니다. 일반적으로 더 좋은 크로모좀일수록 더 높은 값이 나오도록 함수를 설정합니다. 두 크로모좀이 동일한 변수 개수를 가지고 있을 때에는 더 높은 함숫값을 가진 크로모좀이 선택되어야 하며, 두 크로모좀의 함숫값이 같을 경우에는 더 적은 변수를 가지는 크로모좀을 선택하도록 합니다. 일반적으로 다항 선형 회귀에서는 조정된 결정계수(Adjusted R-square), AIC(Akaike information criterion), BIC(Bayesian information criterion) 등이 있습니다.

각 크로모좀에 대한 수치가 나온 뒤에 처리하는 방법에는 순위(Rank)와 가중치(Weight)의 두 가지 방법이 있습니다. 이 두 방법은 이후 선택(Selection)에서 어떤 방식을 사용할 지에 따라 달라지게 됩니다. 순위 방식에서는 단순하게 더 성능이 좋은, 즉 더 값이 높은 크로모좀의 등수를 매깁니다. 가중치 방식에서는 각각의 크로모좀이 가지는 수치값을 전체 크로모좀의 수치 합으로 나눈 값을 사용합니다. 

아래는 조정된 결정계수를 사용하여 각 크로모좀의 평가값을 매긴 뒤에 순위 방식과 가중치 방식을 사용하여 처리해준 것입니다.

<img src="https://user-images.githubusercontent.com/45377884/92330562-c8ab5680-f0aa-11ea-87ad-9fe7f76d70ff.png" alt="fitness_func" style="zoom:50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/Business-Analytics-IME654-">github.com/pilsung-kang/Business-Analytics-IME654</a></p>

가중치 값이 어떻게 구해지는지 1, 2번 크로모좀을 통해서 알아보겠습니다.
$$
\sum_{i=1}^8 (\text{Adj }R^2)_i = 0.75 + 0.78 + \cdots + 0.55 = 4.23 \\
w_1 = 0.75/4.23 = 0.177 \qquad w_2 = 0.78/4.23 = 0.184
$$

## Selection

**선택(Selection)**은 어떤 크로모좀을 선택할 것인지를 결정합니다. 위에서 어떤 방식으로 처리해주었는지에 따라서 선택 방식이 갈리게 됩니다. 순위 방식으로 처리하면 결정론적 선택(Deterministic selection)을 사용합니다. 이 방법은 상위 $N\%$ 만 다음 세대로 넘겨주는 방식으로 $N=25$이면 8개의 크로모좀 중 2개의 크로모좀만 선택하게 됩니다. 방법 자체는 매우 간단하지만, 상대적으로 열등한 축에 속하는 크로모좀은 선택되지 못한다는 단점을 가지고 있습니다.

가중치 방식으로 처리하면 확률론적 선택(Probabilistic selection)을 사용합니다. 이 방법은 가중치 비율만큼의 면적을 가진 원판에 다트를 던져 다트가 꽂히는 크로모좀을 선택하는 방식입니다. 아래의 그림을 보며 확률론적 선택법에 대해 알아보겠습니다.

![prop_selection](https://user-images.githubusercontent.com/45377884/92330986-0eb5e980-f0ae-11ea-952f-ca180e42f6d2.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/Business-Analytics-IME654-">github.com/pilsung-kang/Business-Analytics-IME654</a></p>

먼저 가중치만큼의 범위를 쌓아가면서 $[0,1]$의 누적 가중치 범위를 설정합니다. 그리고 선택하고 싶은 크로모좀 갯수만큼 $[0,1]$범위의 난수를 생성합니다. 위에서는 2개의 크로모좀을 선택하기 위해서 2개의 난수를 생성하였습니다. 그리고 해당 난수가 속하는 범위의 크로모좀을 선택하면 됩니다. 이 방법은 다소 복잡하지만 모든 크로모좀을 선택 후보에 놓을 수 있다는 장점을 가지고 있습니다.

비록 $C_6, C_7$ 등이 선택될 확률 $P(C_6), P(C_7)$은 성능이 좋은 크로모좀인 $C_1, C_2$ 등이 채택될 확률 $P(C_1), P(C_2)$ 보다는 낮지만 일정량의 확률은 보장됩니다. 

## Crossover & Mutation

다음은 번식이 일어나는 **교배(Crossover)** 단계입니다. 이 단계에서는 하이퍼파리미터로 교배 지점(Crossover points)의 개수를 지정해주어야 합니다. 교배 지점은 두 크로모좀의 Gene 값을 바꾸는 기준이 되는 점입니다. 이 점을 기준으로 한 쪽은 값을 바꾸게 되며, 다른 한 쪽은 값을 고정하게 됩니다. 특성의 개수가 $N$일 때, 교배 지점의 개수는 $1 \sim N$까지 설정할 수 있습니다.

교배 지점의 개수를 1로 설정했을 때는 다음과 같이 교배가 수행됩니다.

<img src="https://user-images.githubusercontent.com/45377884/92331237-3f971e00-f0b0-11ea-886b-9938901eb422.png" alt="crossover_1" style="zoom: 50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/Business-Analytics-IME654-">github.com/pilsung-kang/Business-Analytics-IME654</a></p>

교배 지점의 개수를 2로 설정했을 때는 다음과 같이 교배가 수행됩니다.

<img src="https://user-images.githubusercontent.com/45377884/92331241-40c84b00-f0b0-11ea-8f00-1cc46331133a.png" alt="crossover_2" style="zoom:50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/Business-Analytics-IME654-">github.com/pilsung-kang/Business-Analytics-IME654</a></p>

교배 지점의 개수를 N으로 설정했을 때는 다음과 같이 교배가 수행됩니다. 이 때는 모든 Gene이 교배의 대상이 되므로 특성마다 난수를 생성하여 Cut-off 값 이상인 Gene만 교배를 수행합니다. 교배 지점의 개수가 많아질수록 자유도가 높아지는 것이 특징이나 더 높은 자유도가 더 높은 성능을 의미하지는 않습니다.

<img src="https://user-images.githubusercontent.com/45377884/92331242-4160e180-f0b0-11ea-866d-d0d024b15a31.png" alt="crossover_n" style="zoom:50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/Business-Analytics-IME654-">github.com/pilsung-kang/Business-Analytics-IME654</a></p>

다음은 **돌연변이(Mutation)**에 대해 알아보겠습니다. 돌연변이를 만들어주는 이유는 알고리즘이 지역 최적점(Local optimum)으로 다가갈 때 빠져나올 기회를 주기 위해서입니다. 하지만 돌연변이의 비율을 높게 설정하면 수렴 시간이 늘어난다는 단점이 있습니다. 그렇기 때문에 일반적으로는 $0.01$ 정도의 값을 사용합니다.

생성되는 각각의 자식 크로모좀의 Gene마다 난수를 부여한 뒤 설정한 값(대개 $0.01$) 이하의 난수가 나온 경우에만 해당 Gene의 값을 바꾸어줍니다. 이 과정을 그림으로 나타내면 다음과 같습니다.

<img src="https://user-images.githubusercontent.com/45377884/92331480-dc0df000-f0b1-11ea-8aa1-eefd3585517c.png" alt="mutation" style="zoom: 50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/Business-Analytics-IME654-">github.com/pilsung-kang/Business-Analytics-IME654</a></p>

## Find the Best Solution 

마지막으로 최적의 특성을 선택하는 방법입니다. 종료 기준(Stop criteria)을 만족하는 경우에 가장 최적의 크로모좀을 사용하여 특성을 골라냅니다. 반복 과정에서 해 수렴의 안정성을 위해서 한 가지 트릭을 사용합니다. 바로 탑 2를 다음 세대에서 바꾸지 않는 방법입니다. 만약 Population이 100일 때, 이전 세대애서 $C_9, C_{72}$가 가장 높은 성능을 보였다면 다음 세대에서 자식 크로모좀과 바꾸지 않고 그대로 사용합니다. 만약 자식 세대에서 이 둘을 뛰어넘는 성능의 크로모좀이 나오지 않않으면 자신의 자리를 계속 유지하게 됩니다.