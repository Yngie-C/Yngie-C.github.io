---
layout: post
title: 공분산(Covariance)과 상관계수(Correlation Coefficient)
category: Statistics
tag: Statistics
---



이 게시물은 [부산대학교 김충락 교수님의 수리통계학 강의](http://kocw.net/home/search/kemView.do?kemId=1363783)를 참고하여 작성하였습니다.

# Covariance & Correlation

확률 변수가 2개 이상인 경우 그들끼리의 **공분산(Covariance)**를 구할 수 있고, 공분산을 활용하면 변수끼리의 상관성을 나타내는 **상관계수(Correlation coefficient)**를 구할 수 있습니다.

### Covariance

먼저 공분산에 대해서 알아보겠습니다. 두 확률변수 $X_1, X_2$ 의 공분산 $\text{Cov}(X_1, X_2)$ 은 다음의 식을 통해 구할 수 있습니다. 여기서 확률 변수 $X_1, X_2$ 의 기댓값 $E(X_1), E(X_2)$ 은 각각 $\mu_1, \mu_2$ 라고 하겠습니다.


$$
\text{Cov}(X_1, X_2) = E[(X_1-\mu_1)(X_2-\mu_2)]
$$


위 식을 정리하여 아래와 같이 나타낼 수도 있습니다.


$$
\begin{aligned}
E[(X_1-\mu_1)(X_2-\mu_2)]
&= E[(X_1X_2-\mu_1X_2 - \mu_2X_1 +\mu_1\mu_2)] \\
&= E(X_1X_2) - \mu_1E(X_2) - \mu_2E(X_1) + \mu_1\mu_2 \\
&= E(X_1X_2) - \mu_1\mu_2 - \mu_2\mu_1 + \mu_1\mu_2 \\
&= E(X_1X_2) - \mu_1\mu_2 \\
&= E(X_1X_2) - E(X_1)E(X_2)
\end{aligned}
$$



### Correlation Coefficient

다음은 상관관계를 나타내는 상관 계수에 대해서 알아보도록 하겠습니다. 상관 계수는 아래 식과 같습니다.


$$
\rho = \frac{\text{Cov}(X_1, X_2)}{\sigma_1\sigma_2} = \frac{E[(X_1-\mu_1)(X_2-\mu_2)]}{\big\{E[(X_1-\mu_1)^2]E[(X_2-\mu_2)^2]\big\}^{1/2}}
$$


공분산은 각 확률변수에 따라 나오는 값의 범위가 크게 달라지지만 상관계수는 그 값을 각 확률 변수의 표준편차 $\sigma_1,\sigma_2$ 로 나누어 주기 때문에 범위를 $[-1,1]$ 로 좁힐 수 있습니다. 이 값이 $0$ 보다 큰 경우에는 두 확률 변수가 양의 상관관계를 가진다고 하며, 작은 경우에는 음의 상관관계를 가진다고 말합니다.



## Theorem

공분산 및 상관 계수와 관련된 정리에 대해서 알아보겠습니다. 아래 두 정리(1,2)는 조건부 확률의 기댓값 $E(X_2 \vert X_1)$ 이 $X_1$ 에 대해 선형일 때, 즉 $E(X_2 \vert X_1) = a + b \cdot X_1$ 일 때 성립하는 정리입니다.

### Theorem 1

첫 번째 정리는 상관계수를 통해 $E(X_2 \vert X_1)$ 값을 나타내는 정리이며, 수식으로 나타내면 아래와 같습니다.


$$
E(X_2 \vert X_1) = \mu_2 + \rho \cdot \frac{\sigma_2}{\sigma_1}(X-\mu_1)
$$


위 식이 도출되는 과정을 알아보겠습니다. 첫 번째로 아래와 같은 식이 필요합니다.


$$
\begin{aligned}
&E[E(X_2 \vert X_1)] = a + b \cdot E(X_1) = a + b \cdot \mu_1 = \mu_2\\
\because &E[E(X_2 \vert X_1)] = E(X_2) \quad \text{(Double Expectation)}
\end{aligned}
$$


두 번째로는 아래와 같은 식이 필요합니다.


$$
\begin{aligned}
X_1 \cdot E(X_2 \vert X_1) &= E(X_1X_2 \vert X_1) = a \cdot X_1 + b \cdot {X_1}^2 \\
\therefore E[E(X_1X_2 \vert X_1)] &= E(a \cdot X_1 + b \cdot {X_1}^2) \\
&= a \cdot E(X_1) + b \cdot E({X_1}^2) \\
&= a \cdot \mu_1 + b \cdot ({\sigma_1}^2 + {\mu_1}^2) \\
&= E(X_1X_2) \quad \because \text{Double Expectation}
\end{aligned}
$$


마지막으로 아래와 같은 식이 필요합니다. 첫 번째로 도출한 식을 대입하여 


$$
\begin{aligned}
\text{Cov}(X_1, X_2) &= E(X_1X_2) - \mu_1\mu_2 \\
\therefore E(X_1X_2) &= \rho \sigma_1\sigma_2 + \mu_1\mu_2 \\
&= \rho \sigma_1\sigma_2 + \mu_1(a + b \cdot \mu_1)\\
&= a \cdot \mu_1 + b \cdot {\mu_1}^2 + \rho \sigma_1\sigma_2 \\
\because \text{Cov}(X_1, X_2) &= \rho\sigma_1\sigma_2
\end{aligned}
$$


위에서 $E(X_1X_2)$ 에 해당하는 두 식을 연결하면 $b$ 를 구할 수 있습니다. 


$$
\begin{aligned}
a \cdot \mu_1 + b \cdot ({\sigma_1}^2 + {\mu_1}^2) &= a \cdot \mu_1 + b \cdot {\mu_1}^2 + \rho \sigma_1\sigma_2 \\
b \cdot {\sigma_1}^2 &= \rho \sigma_1\sigma_2
\end{aligned}\\
\therefore b = \rho \cdot \frac{\sigma_2}{\sigma_1}
$$


그리고 이 값을 첫 번째 식에 대입하면 $a$ 도 구해낼 수 있습니다.


$$
\begin{aligned}
\mu_2 &= a + b \cdot \mu_1 \\
&= a + \rho \cdot \frac{\sigma_2}{\sigma_1} \cdot \mu_1 \\
\therefore a &= \mu_2 - \rho \cdot \frac{\sigma_2}{\sigma_1} \cdot \mu_1
\end{aligned}
$$


두 값을 $E(X_2 \vert X_1) = a + b \cdot X_1$ 에 대입하여 정리의 식을 구해낼 수 있습니다.


$$
\begin{aligned}
E(X_2 \vert X_1) &= a + b \cdot X_1 \\
&= \bigg(\mu_2 - \rho \cdot \frac{\sigma_2}{\sigma_1} \cdot \mu_1\bigg) + \rho \cdot \frac{\sigma_2}{\sigma_1} \cdot X_1 \\
\therefore E(X_2 \vert X_1)&= \mu_2 + \rho \cdot \frac{\sigma_2}{\sigma_1}(X_1-\mu_1)
\end{aligned}
$$


### Theorem 2

두 번째 정리는 조건부 확률의 분산에 대한 기댓값을 상관 계수로 나타내는 정리입니다. 다음과 같은 수식으로 나타낼 수 있습니다.


$$
E[Var(X_2 \vert X_1)] = {\sigma_2}^2(1 - \rho^2)
$$

수식은 간단하지만 증명하는 과정은 매우 복잡합니다. 그렇지만 하나하나 증명해보도록 하겠습니다. 먼저, 조건부 확률 변수의 분산의 정의에 따라서 $E[Var(X_2 \vert X_1)] = E[\big\{X_2 - E(X_2 \vert X_1)\big\}^2 \vert X_1]$ 이므로 이를 아래와 같이 변형할 수 있습니다.



$$
\begin{aligned}
E[\big\{X_2 - E(X_2 \vert X_1)\big\}^2 \vert X_1] &= \iint \big\{x_2 - \big[\mu_2 + \rho \cdot \frac{\sigma_2}{\sigma_1}(x_1-\mu_1)\big]\big\}^2 f_{X_1,X_2}(x_1, x_2)dx_2dx_1 \\
&= \iint \big\{(x_2 - \mu_2) - (\rho \cdot \frac{\sigma_2}{\sigma_1}(x_1-\mu_1))\big\}^2 f_{X_1,X_2}(x_1, x_2)dx_2dx_1 \\
& = \iint \big\{(x_2 - \mu_2)^2 + \rho^2 \cdot (\frac{\sigma_2}{\sigma_1})^2 \cdot (x_1-\mu_1)^2 - 2\rho \cdot \frac{\sigma_2}{\sigma_1}(x_2 - \mu_2)(x_1-\mu_1)\big\} f_{X_1,X_2}(x_1, x_2)dx_2dx_1 \\
&= {\sigma_2}^2 +  \rho^2 \cdot (\frac{\sigma_2}{\sigma_1})^2 \cdot {\sigma_1}^2 - 2\rho \cdot \frac{\sigma_2}{\sigma_1} \cdot \text{Cov}(X_1, X_2) \\
&= {\sigma_2}^2 +  \rho^2 \cdot (\frac{\sigma_2}{\sigma_1})^2 \cdot {\sigma_1}^2 - 2\rho \cdot \frac{\sigma_2}{\sigma_1} \cdot \rho\sigma_2\sigma_1 \\
&= {\sigma_2}^2 +  \rho^2 \cdot {\sigma_2}^2 - 2\rho^2 \cdot {\sigma_2}^2 \\
&= {\sigma_2}^2 (1-\rho^2)
\end{aligned}
$$



