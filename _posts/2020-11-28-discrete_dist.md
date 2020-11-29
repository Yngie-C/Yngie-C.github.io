---
layout: post
title: 이항 확률 분포(Binomial Distribution)와 친구들
category: Statistics
tag: Statistics
---



이 게시물은 [부산대학교 김충락 교수님의 수리통계학 강의](http://kocw.net/home/search/kemView.do?kemId=1363783)를 참고하여 작성하였습니다.

# Binomial Distribution

이번 시간에는 **이항 분포(Binomial distribution)**와 그와 관련된 분포들에 대해서 알아보도록 하겠습니다. 아래 이미지에서 빨간색 박스가 쳐진 4가지 분포에 대해서 알아보겠습니다.

 

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/100543119-a718d000-3291-11eb-9ee5-ee86e55c0ab5.png" alt="image (2)" style="zoom:80%;" /></p>



## Bernoulli Trial

먼저 **베르누이 시행(Bernoulli trial)**에 대해서 알아보겠습니다. 베르누이 시행이란 다음의 세 가지 조건을 만족하는 실험입니다.

> 1. 베르누이 시행의 **결과는** 성공(Success)과 실패(Failure) $\{S,F\}$ 의 **두 가지**로 나타납니다.
> 2. 두 번째로 각 시행은 서로 **독립**(independent)입니다.
> 3. 성공 확률 $P(S)$ 이 **일정**(Constant)합니다.

예를 들면, 동전을 던져서 앞면이 나오는 경우를 성공이라고 하는 경우가 있습니다. 성공과 실패 이외에 다른 결과가 나오지 않고 매 시행이 독립이며 성공 확률이 변하지 않으므로 베르누이 시도입니다. 비슷한 예시로, 주사위를 던져서 3이상의 눈이 나오는 경우를 성공이라고 정의한다면 베르누이 시행이 됩니다.

한 번의 베르누이 시행은 $B(1,p)$ 로 나타냅니다. 괄호 안의 숫자 중 앞에 있는 것은 베르누이 시행의 횟수이며 뒤에 해당하는 $p$ 는 성공 확률 $P(S)$ 입니다. 베르누이 시행에서 성공과 실패를 나타내는 확률 변수 $X$ 에 대한 확률 질량 함수는 다음과 같습니다.



$$
P_X(x) = p^x(1-p)^{1-x} \qquad I(x = 0,1)
$$



확률 질량 함수로부터 베르누이 시행의 기댓값과 분산을 구할 수 있습니다.



$$
\begin{align}
E(X) &= \sum_{x=0}^1 x \cdot P_X(x) \\
&= \sum_{x=0}^1 x \cdot p^x(1-p)^{1-x} \\
&= 0 \cdot p^0 \cdot (1-p)^1 + 1 \cdot p^1 \cdot (1-p)^0 = p
\end{align}
$$



분산은 $Var(X) = E(X^2) - [E(X)]^2$ 를 활용하여 구할 수 있습니다.



$$
\begin{align}
Var(X) &= E(X^2) - [E(X)]^2 \\
&= \sum_{x=0}^1 x^2 \cdot p^x(1-p)^{1-x} - p^2 \\
&= \sum_{x=0}^1 x^2 \cdot p^x(1-p)^{1-x} - p^2 \\
&= 0 \cdot p^0(1-p)^1 + 1 \cdot p^1(1-p)^0 - p^2 \\
&= p - p^2 = p(1-p)
\end{align}
$$



## Binomial Distribution

다음으로 이항 분포에 대해서 알아보겠습니다. 이항 분포는 베르누이 시행을 여러 번 진행했을 때 성공 횟수를 나타내는 확률 변수에 대한 분포입니다. 각 베르누이 시행을 $Y_1, \cdots, Y_n \quad (Y = 0,1)$ 이라 두고 성공 횟수의 합을 확률 변수 $X$, 즉 $X := \sum^n_{i=1} Y_i$ 이라 하고 $X$ 의 확률 질량 함수를 구해보겠습니다.


$$
P_X(x) = \left(\begin{array}{ccc} n \\ x\end{array} \right) p^x(1-p)^{n-x}
$$


이항 분포는 $X \sim B(n,p)$ 로 표기할 수 있으며 확률 질량 함수를 알았으니, 이를 활용하여 적률 생성 함수를 구할 수 있습니다.



$$
\begin{align}
M_X(t) &= E(e^{tX}) \\
&= \sum^n_{x=0} e^{tx} \cdot P_X(x) \\
&= \sum^n_{x=0} e^{tx} \cdot \left(\begin{array}{ccc} n \\ x\end{array} \right) p^x(1-p)^{n-x} \\
&= \sum^n_{x=0} \left(\begin{array}{ccc} n \\ x\end{array} \right) (p\cdot e^t)^x(1-p)^{n-x} \\
&= [(1-p) + p\cdot e^t]^n
\end{align}
$$



적률 생성 함수를 활용하여 이항 분포의 기댓값과 분산도 구해볼 수 있습니다. 먼저 기댓값 $E(X)$ 은 1차 모멘트와 같고, 이는 적률 생성 함수를 한 번 미분한 함수에 $0$ 을 대입하여 구할 수 있으므로 아래와 같습니다. 계산 과정은 생략하겠습니다.


$$
\begin{align}
M^\prime(t) &= np\cdot e^t \cdot [(1-p) + p\cdot e^t]^{n-1} \\
\therefore M^\prime(0) &= np = E(X) 
\end{align}
$$


분산을 구하기 위해서는 2차 모멘트 $E(X^2)$ 를 알아야 합니다. 2차 모멘트는 적률 생성 함수를 2번 미분한 함수에 $0$ 을 대입하여 구할 수 있습니다.


$$
\begin{align}
M^{\prime\prime}(t) &= np\cdot e^t \cdot [(1-p) + p\cdot e^t]^{n-2}(np\cdot e^t+1-p) \\
M^{\prime\prime}(0) &= np(np+1-p) = E(X^2)
\end{align}
$$


분산 $Var(X)$ 는 1,2차 모멘트의 결합인 $Var(X) = E(X^2) - E(X)^2$ 로 나타낼 수 있으므로 아래와 같이 구해집니다.


$$
\begin{align}
Var(X) &= E(X^2) - E(X)^2 \\
&= np(np+1-p) - (np)^2 \\
&= np(1-p)
\end{align}
$$


### Theorem

다음은 이항 분포에서 성립하는 정리에 대해서 알아보겠습니다. 이항 분포를 만족하는 확률 변수 $Y$ 에 대해서 다음의 식이 만족합니다.


$$
\text{if} \quad n \rightarrow \infty, \quad
P\bigg[\bigg\vert \frac{Y}{n} - p \bigg\vert \geq \varepsilon \bigg] \rightarrow 0
$$


위 식에서 $Y$ 는 $n$ 번의 베르누이 시행 중에서 성공한 횟수이므로 $\frac{Y}{n}$ 은 표본 사건의 성공 비율이라고 할 수 있습니다. $p$ 는 모집단에서 베르누이 시행의 성공 확률입니다. 따라서 위 정리는 표본의 개수가 많아질 때 표본 사건의 성공 비율과 모집단의 성공 비율의 차이가 $0$ 에 가까워진다는 것이지요. 위 정리를 **약대수의 법칙(Weak Law of Large Numbers, WLLN)** 이라고 합니다. 그리고 아래와 같이 체비쇼프 부등식(Chebyshev inequality)으로부터 증명할 수 있습니다.

체비쇼프 부등식은 $E(X) = \mu, Var(X) = \sigma^2$ 인 확률 변수 $X$ 에 대하여 아래와 같습니다. 

 
$$
P(\vert X - \mu \vert \geq k \cdot \sigma) \leq \frac{1}{k^2}
$$


정리에서 나타나는 식을 체비쇼프 부등식의 형태로 변형시켜 보겠습니다. $E(Y) = np, Var(Y) = np(1-p)$ 이므로 아래와 같이 식을 변형하여 $k$ 를 구할 수 있습니다.


$$
\begin{align}
P\bigg[\bigg\vert \frac{Y}{n} - p \bigg\vert \geq \varepsilon \bigg] &= P\bigg[\bigg\vert Y - np \bigg\vert \geq n\varepsilon \bigg] \\
&= P\bigg[\bigg\vert Y - np \bigg\vert \geq \frac{n\varepsilon}{\sqrt{np(1-p)}} \cdot \sqrt{np(1-p)} \bigg] \\
\therefore k &= \sqrt{\frac{n}{p(1-p)}} \cdot \varepsilon
\end{align}
$$


$k$ 를 구했으니 체비쇼프 부등식을 활용하여 확률의 범위를 제한할 수 있습니다.


$$
P\bigg[\bigg\vert \frac{Y}{n} - p \bigg\vert \geq \varepsilon \bigg] \leq \frac{p(1-p)}{n\varepsilon^2}
$$


위 식에서 $p, \varepsilon$ 은 상수이므로 $n$ 이 커질수록 확률 $P$ 는 0으로 수렴하게 됩니다.



## Negative Binomial Distribution

이번에는 이항 분포와 같이 여러 번의 베르누이 시행에서 도출할 수 있는 **음이항 분포(Negative binomial distribution)**에 대해 알아보겠습니다. 음이항 분포란 연속적인 베르누이 시행에서 $r$ 번의 성공을 달성하기 까지 몇 번의 실패가 있어야 하는 지에 대한 분포입니다. 일반적인 이항 분포에서는 성공 횟수를 다루었다면 음이항 분포는 실패 횟수를 다룬다는 차이점이 있습니다.

중요한 것은 $r$ 번째의 성공은 가장 마지막에 위치해야 한다는 점입니다. 따라서 확률 질량 함수를 구하기 위해서 $y+r-1$ 번의 시도 중 실패 횟수가 $y$ 이고 성공 횟수가 $r-1$ 인 이항 분포를 구한 뒤, 마지막으로 성공한 베르누이 시행을 곱합니다. 확률 질량 함수는 아래와 같이 구할 수 있습니다. $Y \sim NB(r,p)$ 로 표기합니다.


$$
\begin{align}
P_Y(y) &= \left(\begin{array}{ccc} y+r-1 \\ y\end{array} \right) p^{r-1}(1-p)^y \cdot p \\
\therefore P_Y(y) &= \left(\begin{array}{ccc} y+r-1 \\ y\end{array} \right) p^r(1-p)^y
\end{align}
$$


확률 질량 함수를 구했으니 적률 생성 함수를 구할 수 있습니다. 이항 분포와 달리 음이항 분포는 횟수가 정해진 것이 아니므로, 실패 횟수를 나타내는 확률 변수 $Y$ 가 가질 수 있는 범위는 $Y = 0,1,2, \cdots$ 라는 점에 유의합니다.


$$
\begin{align}
M_Y(t) &= E[e^{tY}] \\
&= \sum^\infty_{y=0} e^{ty} \cdot \left(\begin{array}{ccc} y+r-1 \\ y\end{array} \right) p^r(1-p)^y \\
&= p^r \cdot \sum^\infty_{y=0} \left(\begin{array}{ccc} y+r-1 \\ y\end{array} \right) [(1-p)e^t]^y \\
&= p^r \cdot \sum^\infty_{y=0} \frac{(y+r-1)!}{y!(r-1)!} [(1-p)e^t]^y \\
&= p^r \cdot \bigg(1 + r \cdot [(1-p)e^t] + \frac{r(r+1)}{2}[(1-p)e^t]^2 + \cdots \bigg) \\
&= p^r \cdot \bigg(1 + r \cdot T + \frac{r(r+1)}{2}T^2 + \cdots \bigg) \quad T := (1-p)e^t
\end{align}
$$


식이 상당히 복잡합니다. 식을 간단하게 정리하기 위해서 $g(t) = (1-t)^{-r}$ 이라는 함수를 생각해보겠습니다. 이 함수를 테일러 급수를 이용하여 전개하면 위와 동일한 형태의 식이 나오게 됩니다.


$$
\begin{align}
g(t) &= g(0) + g^\prime(0) \cdot t + \frac{g^{\prime\prime}(0)}{2} \cdot t^2 + \cdots \\
&= 1 + r \cdot t + \frac{r(r+1)}{2} \cdot t^2 + \cdots
\end{align}
$$


이를 이용하면 적률 생성 함수를 간단히 정리할 수 있게 됩니다.


$$
M_Y(t) = p^r \cdot [1-(1-p)e^t]^{-r}
$$


이를 활용하여 음이항 분포 $NB(r,p)$ 의 기댓값과 분산을 구하면 아래와 같습니다.


$$
\begin{align}
E(Y) &= M_Y^\prime(0) = \frac{r(1-p)}{p} \\
Var(Y) &= M_Y^{\prime\prime}(0) - M_Y^\prime(0)^2 = \frac{r(1-p)}{p^2}
\end{align}
$$


## Geometric Distribution

**기하 분포(Geometric distribution)**는 음이항 분포의 특수한 형태입니다. 첫 번째 성공이 있을 때까지 몇 번의 실패가 있을 지를 구하는 문제이지요. 음이항 분포에서 성공 횟수를 나타내는 $r$ 이 1인 형태, 즉 $NB(1,p)$ 라고 할 수 있겠습니다. $r$ 이 1로 고정되기 때문에 파라미터는 오직 $p$ 하나가 됩니다. 따라서, 기하 분포는 $G(p)$ 또는 $Geo(p)$ 로 나타낼 수 있습니다.

확률 질량 함수와 적률 생성 함수도 음이항 분포의 것을 활용하여 구할 수 있습니다.


$$
\begin{aligned}
P_Y(y) &= p\cdot (1-p)^y \\
M_Y(t) &= \frac{p}{1-(1-p)e^t}
\end{aligned}
$$
