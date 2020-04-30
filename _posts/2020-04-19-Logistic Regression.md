---
layout: post
title: 로지스틱 회귀 (Logistic Regression)
category: Machine Learning
tag: Machine-Learning
---

 본 포스트는 [카이스트 문일철 교수님의 강의](https://www.edwith.org/machinelearning1_17/joinLectures/9738) 를 바탕으로 작성하였습니다.



# Logistic Regression

[이전 게시물]([https://yngie-c.github.io/machine%20learning/2020/04/08/naive_bayes/](https://yngie-c.github.io/machine learning/2020/04/08/naive_bayes/)) 에서는 나이브 베이즈 분류기(Naive Bayes Classifier)에 대해 알아보았다. 나이브 베이즈 분류기가 가진 가장 큰 문제는 말 그대로 **Naive Assumption** (모든 특성이 조건부 독립이라는 가정)이었다. 이번 시간에 등장하는 로지스틱 회귀(Logistic Regression)는 이런 가정 없이 데이터를 분류할 수 있는 방법 중 하나다.



## 결정 경계 (Decision Boundary)

특정한 X를 기점으로 분류기가 판단하는 클래스가 바뀔 때 그 X를 결정 경계라고 한다. 베이즈 분류기에서 알아본 것처럼 결정 경계 부근에서 확률이 급격하게 변할수록 더 좋은 분류기이다. 분류기의 Error라고 할 수 있는 Bayes risk가 줄어들기 때문이다. 대신에 분류기를 어떻게 설계하든 확률값을 내놓아야 하므로 최댓값이 1을 넘지 않고 최솟값이 0보다 작지 않아야 한다. 이런 모양을 가지는 대표적인 함수가 시그모이드(Sigmoid) 함수이다.



## 시그모이드 (Sigmoid)

**시그모이드 함수(Sigmoid Function)** 란 실수 정의역 범위에서 최댓값이 1이하이고 최솟값이 0이상인 미분가능한 단조증가함수를 일컫는다. 시그모이드 함수가 하나로 정해져 있는 것은 아니다. $\tanh{x} , \arctan{x} , 1/(1+e^{-x})$ 등 다양한 형태의 시그모이드 함수가 있다. 세번째에 있는 $1/(1+e^{-x})$ 함수는 **로지스틱(Logistic) 함수** 라 일컫는다. 이번 시간에는 많은 시그모이드 중 로지스틱 함수에 대해서 알아볼 것이다.

로지스틱 함수를 사용하는 이유는 간단하다. 미분 계산이 쉽기 때문이다. 최적화(Optimization) 하는 과정에서 미분 계산은 당연히 필요한 관문이다. 로지스틱은 이를 간단히 만들어주기 때문에 데이터셋이 많아지더라도 시간과 컴퓨팅 자원을 적절한 선에서 유지할 수 있다.

로지스틱 함수가 생성되는 과정은 아래와 같다. 먼저 로짓(Logit)이라는 로지스틱 함수의 역함수에서 시작한다. 각각의 조건부 확률 $P(Y|X)$ 로부터의 모델이 필요하므로 역함수를 취하여 $x$ 에 관한 식으로 정리해준다. 그 다음 임의의 파라미터 $a, b$ 를 추가하여 모델을 데이터에 좀 더 잘 맞출 수 있다. 그리고 이를 특성 행렬 $X$ 와 파라미터 행렬 $\theta$ 로 나타낼 수 있다. 



$$
f(x)=\log(\frac{x}{1-x}) \rightarrow x = \log(\frac{p}{1-p}) \rightarrow ax+b = \log(\frac{p}{1-p}) \rightarrow X\theta = \log(\frac{p}{1-p})
$$



최대 우도 추정(Maximum Likelihood Estimation, MLE) $\hat{\theta} = \text{argmax}_{\theta} P(D|\theta)$ 으로부터 최대 조건부 우도 추정 MCLE(Maximum Conditional Likelihood Estimation)을 유도할 수 있다. 아래의 식을 보자.



$$
\hat{\theta} = \text{argmax}_{\theta} P(D|\theta) = \text{argmax}_{\theta} \prod_{1 \leq i \leq N} P(Y_i|X_i;\theta) \qquad \quad \\
\qquad \qquad \qquad = \text{argmax}_{\theta} \log (\prod_{1 \leq i \leq N} P(Y_i|X_i;\theta)) = \text{argmax}_{\theta} \sum_{1 \leq i \leq N} \log (P(Y_i|X_i;\theta))
$$



베르누이 시행 $P(y|x) = \mu(x)^y (1-\mu(x))^{1-y}$ 에서 $P(Y_i|X_i;\theta) = \mu(X_i)^{Y_i} (1-\mu(X_i))^{1-Y_i}$ 의 식을 유도할 수 있다. 이 식에 로그함수를 취하면 다음과 같이 식을 변형할 수 있다.



$$
\log P(Y_i|X_i;\theta) = Y_i \log \mu(X_i) + (1-Y_i) \log (1-\mu(X_i)) \qquad \qquad \qquad \qquad \qquad \quad \\
= Y_i\{\log \mu(X_i) - \log (1-\mu(X_i))\} + \log (1-\mu(X_i)) \\
= Y_i \log \frac{\mu(X_i)}{(1-\mu(X_i))} + \log (1-\mu(X_i)) \qquad \qquad \quad \\
= Y_i X_i \theta + \log (1-\mu(X_i)) = Y_i X_i \theta - \log (1+e^{X_i\theta})
$$



이 식을 위에서 구했던 $\text{argmax}_\theta$ 식에 대입한 후 $\theta$ 로 편미분하여 0이 되는 점을 구해볼 수 있다.



$$
\hat{\theta} = \text{argmax}_{\theta} \sum_{1 \leq i \leq N} \{Y_i X_i \theta - \log (1+e^{X_i\theta})\} \\
\frac{\partial}{\partial \theta_j}\{\sum_{ 1 \leq i \leq N} Y_i X_i \theta - \log (1+e^{X_i\theta})\}
= \sum_{ 1 \leq i \leq N} Y_i X_{i,j} + \sum_{ 1 \leq i \leq N} - \frac{1}{1+e^{X_i\theta}} \times e^{X_i\theta} \times X_{i,j} \\
= \sum_{ 1 \leq i \leq N} X_{i,j}(Y_i - \frac{e^{X_i \theta}}{1+e^{X_i\theta}})
= \sum_{ 1 \leq i \leq N} X_{i,j}(Y_i - P(Y_i=1|X_i;\theta)) = 0 \qquad \qquad
$$



이렇게 도출된 함수는 로지스틱 함수이기 때문에 쉽게 풀리지 않는다.(보강 해주셈) 그러므로 근사를 통해서 근 $\theta$ 을 추적해나가야 한다.



# Gradient Method

이렇게 근을 근사해나가는 방법 중 하나로 **경사법(Gradient Method)** 이 있다. 경사법으로 **경사 하강법(Gradient Descent Method)과 경사 상승법(Gradient Ascent Method)** 이 있다.

## Taylor Series Expansion

임의의 함수를 다항함수의 합으로 근사하는 테일러 급수 전개(Taylor Series Expansion)로부터 경사법을 알아가보자.



$$
f(x) = f(a) + \frac{f^{\prime}(a)}{1!}(x-a) + \frac{f^{\prime\prime}(a)}{2!}(x-a)^2 + \cdots \\
= \sum^{\infin}_{n=0}\frac{f^{(n)}(a)}{n!}(x-a)^n \qquad \qquad \qquad \qquad
$$



경사 하강/상승법은 주어진 임의의 미분가능한 함수 $f(x)$ 에 대해서 초기 지점 $x_1$ 을 잡는다. $x_1$ 을 기준으로 점차적으로 지점을 움직여 나간다. 해당 지점에서의 기울기를 통해 다음으로 방향을 잡는다. 테일러 급수 전개를 통해 경사법이 작동하는 방식을 살펴보자.



$$
f(x) = f(a) + \frac{f^{\prime}(a)}{1!}(x-a) + O(||x-a||^2) \\
a=x_1 \text{이고 } x=x_1+h\mathbf{u} \quad \mathbf{u} \text{ is unit direction vector} \\
f(x_1+h\mathbf{u}) = f(x_1) + hf^{\prime}(x_1)\mathbf{u} + h^2O(1) \\
f(x_1+h\mathbf{u}) - f(x_1) \sim hf^{\prime}(x_1)\mathbf{u} \qquad \qquad
$$



와 같이 나타낼 수 있다. 경사 하강법의 경우에는 좌변을 가장 작게 만들어주어야 한다. 이 때 유닛벡터 $\mathbf{u}$ 는 다음과 같이 구할 수 있다.



$$
\mathbf{u}^* = \text{argmin}_\mathbf{u}\{f(x_1+h\mathbf{u}) - f(x_1)\} = \text{argmin}_\mathbf{u}hf^{\prime}(x_1)\mathbf{u} = -\frac{f^{\prime}(x_1)}{|f^{\prime}(x_1)|} \\
\because f(x_1+h\mathbf{u}) \leq f(x_1), \quad \vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}| \cos \theta \\
\therefore x_{t+1} \rightarrow x_t + h\mathbf{u}^* = x_t -h\frac{f^{\prime}(x_t)}{|f^{\prime}(x_t)|}
$$



경사 상승법을 사용할 때에는 $\text{argmin}$ 대신 $\text{argmax}$ 를 사용한다. 이렇게 구한 경사 상승법에서의 $x_{t+1}$ 은 다음과 같다.



$$
x_{t+1} \rightarrow x_t + h\mathbf{u}^* = x_t + h\frac{f^{\prime}(x_t)}{|f^{\prime}(x_t)|}
$$



우리가 구해야 하는 것은 파라미터 이므로 $x$ 대신 $\theta$ 를 대입해주면 다음과 같이 식을 쓸 수 있다. (여기서는 경사 상승법에 대해서만 예시를 알아보도록 한다)


$$
\theta_{t+1} \rightarrow \theta_t + h\mathbf{u}^* = \theta_t + h\frac{f^{\prime}(\theta_t)}{|f^{\prime}(\theta_t)|}
$$


$f(\theta)$ 식을 $\theta$ 로 편미분하여 편도함수 식을 구해낼 수 있다.  


$$
f(\theta) = \text{argmax}_{\theta} \sum_{1\leq i \leq N} \log P(Y_i|X_i;\theta) \\
\frac{\partial f(\theta)}{\partial \theta_j} = \sum_{1\leq i \leq N} X_{i,j} (Y_i - P(y=1|x;\theta))
$$


이 식을 



# Gaussian NB vs Logistic Regression

## Gaussian Naive Bayes Classifier

이전에 배운 나이브 베이즈 분류기는 범주형 변수에 대해서만 다루어졌다. 이를 연속형 변수에도 사용할 수 있도록 바꾼 것을 **가우스 나이브 베이즈(Gaussian Naive Bayes)** 분류기라고 한다. 먼저 나이브 베이즈 분류기 함수부터 살펴보도록 하자.


$$
f_{NB}(x) = \text{argmax}_{Y=y} P(Y=y) \prod_{1 \leq i \leq d}P(X_i = x_i | Y = y)
$$


뒤에 오는 조건부 확률에 대한 항을 가우스 정규 분포식으로 대체하고 $P(Y=y)$ 를 상수로 대체하면 다음과 같이 나타낼 수 있다. 


$$
P(Y) \prod_{1 \leq i \leq d}P(X_i|Y) = \pi_k  \prod_{1 \leq i \leq d} \frac{1}{\sigma^i_k C} \exp(-\frac{1}{2}(\frac{x_i - \mu_k^i}{\sigma^i_k})^2)
$$


지금부터의 과정은 Gaussian Naive Bayes 에 몇몇 가정을 추가하여 로지스틱 회귀로 식을 변형하는 과정이다. 첫 번째로는 나이브 베이즈 가정(Naive Bayes Assumption)을 사용하여 식을 변형한다.


$$
P(Y=y|X) = \frac{P(X|Y=y)P(Y=y)}{P(X)} = \frac{P(X|Y=y)P(Y=y)}{P(X|Y=y)P(Y=y) +P(X|Y=n)P(Y=n)} \\
\qquad = \frac{P(Y=y) \prod_{1 \leq i \leq d}P(X_i|Y=y)}{P(Y=y)\prod_{1 \leq i \leq d}P(X_i|Y=y) +P(Y=n)\prod_{1 \leq i \leq d}P(X_i|Y=n)} \\
\qquad = \frac{\pi_1  \prod_{1 \leq i \leq d} \frac{1}{\sigma^i_1 C} \exp(-\frac{1}{2}(\frac{x_i - \mu_1^i}{\sigma^i_1})^2)}{\pi_1  \prod_{1 \leq i \leq d} \frac{1}{\sigma^i_1 C} \exp(-\frac{1}{2}(\frac{x_i - \mu_1^i}{\sigma^i_1})^2) + \pi_2 \prod_{1 \leq i \leq d} \frac{1}{\sigma^i_2 C} \exp(-\frac{1}{2}(\frac{x_i - \mu_2^i}{\sigma^i_2})^2)} \\
= \frac{1}{1 + \frac{\pi_2 \prod_{1 \leq i \leq d} \frac{1}{\sigma^i_2 C} \exp(-\frac{1}{2}(\frac{x_i - \mu_2^i}{\sigma^i_2})^2)}{\pi_1  \prod_{1 \leq i \leq d} \frac{1}{\sigma^i_1 C} \exp(-\frac{1}{2}(\frac{x_i - \mu_1^i}{\sigma^i_1})^2)}} \qquad \qquad \qquad \qquad \qquad \qquad \quad
$$


여기서 등분산 가정 $(\sigma_1 = \sigma_2)$ 을 해주면 식을 좀 더 로지스틱 함수의 형태에 가깝게 나타낼 수 있다.


$$
P(Y=y|X) = \frac{1}{1 + \frac{\pi_2 \prod_{1 \leq i \leq d} \frac{1}{\sigma^i_2 C} \exp(-\frac{1}{2}(\frac{x_i - \mu_2^i}{\sigma^i_2})^2)}{\pi_1  \prod_{1 \leq i \leq d} \frac{1}{\sigma^i_1 C} \exp(-\frac{1}{2}(\frac{x_i - \mu_1^i}{\sigma^i_1})^2)}}
= \frac{1}{1 + \frac{\pi_2 \prod_{1 \leq i \leq d} \exp(-\frac{1}{2}(\frac{x_i - \mu_2^i}{\sigma^i_2})^2)}{\pi_1  \prod_{1 \leq i \leq d} \exp(-\frac{1}{2}(\frac{x_i - \mu_1^i}{\sigma^i_1})^2)}} \qquad \qquad \qquad \quad \\
= \frac{1}{1 + \frac{\pi_2 \exp(-\sum_{1 \leq i \leq d} \{ \frac{1}{2}(\frac{x_i - \mu_2^i}{\sigma^i_2})^2\})}{\pi_1 \exp(-\sum_{1 \leq i \leq d} \{ \frac{1}{2}(\frac{x_i - \mu_1^i}{\sigma^i_1})^2\})}}
= \frac{1}{1 + \frac{\exp(-\sum_{1 \leq i \leq d} \{ \frac{1}{2}(\frac{x_i - \mu_2^i}{\sigma^i_2})^2\} + \log \pi_2)}{\exp(-\sum_{1 \leq i \leq d} \{ \frac{1}{2}(\frac{x_i - \mu_1^i}{\sigma^i_1})^2\} + \log \pi_1)}} \\
\qquad \qquad = \frac{1}{{1 + \exp(-\sum_{1 \leq i \leq d} \{ \frac{1}{2}(\frac{x_i - \mu_2^i}{\sigma^i_2})^2\} + \log \pi_2 +\sum_{1 \leq i \leq d} \{ \frac{1}{2}(\frac{x_i - \mu_1^i}{\sigma^i_1})^2\} - \log \pi_1)}} \\
\qquad = \frac{1}{{1 + \exp(-\frac{1}{2(\sigma_1)^2}\sum_{1 \leq i \leq d} \{(x_i - \mu_1^i)^2 - (x_i - \mu_2^i)^2\}  + \log \pi_2 - \log \pi_1)}} \\
\qquad = \frac{1}{{1 + \exp(-\frac{1}{2(\sigma_1)^2}\sum_{1 \leq i \leq d} \{2(\mu_2^i - \mu_1^i)X_i + {\mu_1^i}^2 - {\mu_2^i}^2\}  + \log \pi_2 - \log \pi_1)}}
$$


로지스틱 회귀를 사용할 경우에는 d+1개의 파라미터가 필요한데 비해 이렇게 구해진 나이브 베이즈 함수의 파라미터 개수는 4d + 1개이다. 게다가 식을 변형하는 과정에서 나이브 베이즈 가정뿐만 아니라 등분산 가정까지 해주었다. 이런 상황을 돌이켜보면 로지스틱 회귀가 나이브 베이즈 분류기보다 더 좋다고 생각할 수도 있다. 하지만 정해진 것은 없다. 주어진 데이터 안에서 알고있는 정보 등에 따라서 적절한 것을 찾는 것이 중요하다. 나이브 베이즈 분류기는 대표적인 **Generative Model** 이고 로지스틱 회귀는 대표적인 **Discriminative Model** 이다. 이 두 모델에 대한 설명은 [Ratsgo님 블로그]([https://ratsgo.github.io/generative%20model/2017/12/17/compare/](https://ratsgo.github.io/generative model/2017/12/17/compare/)) 글을 참조하면 더 자세히 알아볼 수 있다.