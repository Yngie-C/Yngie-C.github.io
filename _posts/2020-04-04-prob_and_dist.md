---
layout: post
title: Probability & Distribution
category: Machine Learning
tag: Machine-Learning
---

 본 포스트는 [카이스트 문일철 교수님의 강의](https://www.edwith.org/machinelearning1_17/joinLectures/9738) 를 바탕으로 작성하였습니다.



# Probability

## Probability

"확률은 ~다."라고 정의하는 것이 쉬운 일은 아니다. 하지만 수학적 확률로 범위를 좁히면 나타낼 수 있다. 수학에서의 확률은 아래의 특징을 가지는 함수로 정의할 수 있다. 모든 사건을 나타내는 $\Omega$ 에 대하여 특정 사건 $E$ 의 확률 $P(E)$ 는
$$
P(E) \in R, \quad P(E) \geq 0, \quad P(\Omega) = 1 \\
P(E_1 \cup E_2 \cup ...) = \sum^{\infty}_{i=1} P(E_i) \quad \text{단, 서로 상호 배타적(Mutually Exclusive)일 경우}
$$
확률의 다른 속성은 아래와 같은 것이 있다.
$$
\text{만약 } A \subset B \text{ 이면 } P(A) \leq P(B), \quad P(\phi) = 0, \quad 0 \leq P(A) \leq 1 \\
P(X \cup Y) = P(X) + P(Y) - P(X \cap Y) \qquad P(E^c) = 1 - P(E)
$$


## Conditional Probability(조건부 확률)

위에서는 아무 조건도 없이 확률을 다루어 보았다. 이번에는 좀 더 범위를 좁혀보자. 범위를 좁히기 위해서 특정한 조건(Condition)을 가미할 수 있다. 예를 들어, '사건 B가 일어났을 때'라는 조건을 추가한다고 해보자. 사건 B가 일어났을 때 사건 A가 일어날 확률은 아래와 같이 나타낼 수 있다. 그리고 이렇게 조건이 있을 때의 확률을 **Conditional Probability(조건부 확률)** 이라고 한다. 
$$
P(A \vert B) = \frac{P(A \cap B)}{P(B)}
$$
위 식을 확장시키면 아래의 식도 도출해 낼 수 있다. 아래 식 중 첫 번째 식은 MAE 관점에서의 확률을 계산하는 데에도 사용되었다.
$$
P(B \vert A) = \frac{P(A \vert B)P(B)}{P(A)} \qquad \text{Ex) } P(\theta \vert D) = \frac{P(D \vert \theta)P(\theta)}{P(D)} \\
P(A) = \sum_n P(A \vert B_n)P(B_n)
$$


# Distribution

## Probability Distribution

**Probability distribution(확률 분포)** 란 확률을 사건에 대해 나타낸 함수이다. 사건은 연속적인 값(Continuous numeric value)일수도 있고 이산적인 값(Discrete categorical value)일수도 있다. 확률 분포에서 $x$ 는 각각 사건을 나타내며 $f(x)$ 는 사건 $x$ 에 할당된 확률을 나타낸다.



## Normal Distribution(정규 분포)

가장 많이 사용되는 분포이다. 사건 $x$ 가 연속적인 값을 나타내는 경우에 쓰인다. $N(\mu, \sigma^2)$ 로 나타낸다.
$$
f(x;\mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x - \mu)^2}{2 \sigma^2}} \\
\mu : \text{mean} \qquad \sigma^2 : \text{variance}
$$


## Beta distribution(베타 분포)

정규 분포와 비슷하지만 사건 $x$ 의 범위가 정해져 있다는 차이가 있다. (정규 분포는 $x$ 의 범위가 $[-\infty, \infty]$ 기 때문에 long tail이 있다) 베타 분포는 $x$ 의 범위가 $[0,1]$ 기 때문에 확률을 모델링 할 때 자주 사용된다. $\text{Beta} (\alpha, \beta)$ 로 나타낸다.
$$
f(\theta; \alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)} \\
B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}, \quad \Gamma(\alpha) = (\alpha -1)!, \quad \alpha \in N^+ \\
\frac{\alpha}{\alpha + \beta} : \text{mean} \qquad \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta+1)} : \text{variance}
$$


## Binomial distribution(이항 분포)

Bernoulli trial에 대해 사건 $x$가 이산적일(discrete) 경우를 표현하는 가장 간단한 분포이다. $B(n,p)$ 로 나타낸다.

$$
f(x;n, p) = \left(\begin{array}{c}n \\k \end{array}\right)p^k (1-p)^{n-k}, \qquad \left(\begin{array}{c}n \\k \end{array}\right) = \frac{n!}{k!(n-k)!} \\
np : \text{mean} \qquad np(1-p) : \text{variance}
$$


## Multinomial distribution(다항 분포)

이항 분포를 일반화하여 나타낸 확률 분포이다. 2가지 이상의 이산적인 사건을 나타낼 때 사용하며 간단한 예시로 주사위를 던질 때 나오는 특정 숫자가 나오는 경우를 생각해 볼 수 있다. Text generation에서 다음에 올 단어를 예측하는 것도 이산적인 여러 사건 중 하나를 선택하는 경우이기 때문에 다항 분포에 속한다. $\text{Mult}(P), P=<p_1, ... , p_k>$ 로 나타낸다.
$$
f(x_1,x_2 ..., x_k; n,p_1,p_2 ..., p_k) = \frac{n!}{x_1!x_2! ... x_k!}p_1^{x_1}p_2^{x_2}...p_k^{x_k} \\
E(x_i) = np_i : \text{mean} \qquad \text{Var}(x_i) = np_i(1-p_i) : \text{variance}
$$
지금까지 알아본 확률 분포 이외에도 Poisson distribution(푸아송 분포) 등 다양한 확률 분포가 있지만 여기서는 이정도만 알아보도록 한다.

