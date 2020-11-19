---
layout: post
title: 이변수 조건부 확률 분포(Conditional Distribution on bi-r.v s)
category: Statistics
tag: Statistics
---



이 게시물은 [부산대학교 김충락 교수님의 수리통계학 강의](http://kocw.net/home/search/kemView.do?kemId=1363783)를 참고하여 작성하였습니다.

# Conditional Distribution (multi r.v)

이변수 함수의 조건부 확률 질량 함수(Conditional p.m.f.)는 다음과 같습니다.


$$
P_{X_2 \vert X_1}(x_2 \vert x_1) = \frac{P_{X_1,X_2}(x_1,x_2)}{P_{X_1}(x_1)}
$$


만약 $X_1,X_2$ 가 연속형 변수라면 조건부 확률 분포 함수(Conditional p.d.f.)는 다음과 같습니다.


$$
f_{X_2 \vert X_1}(x_2 \vert x_1) = \frac{f_{X_1,X_2}(x_1,x_2)}{f_{X_1}(x_1)}
$$


## Expected Value

조건부 확률에 대한 기댓값(Expected Value)는 어떻게 구할 수 있을까요? 기본적으로는 확률 변수가 1개일 때의 기댓값을 구하는 과정과 동일합니다. 다만 확률 분포 부분에 조건부 확률 분포를 사용합니다.



$$
E[u(X_2) \vert X_1] = \int u(x_2) \cdot f(x_2 \vert x_1) dx_2
$$



조건부 확률의 분산은 기댓값보다는 조금 더 복잡하지만 비슷하게 구할 수 있습니다. 조건부 확률의 분산은 아래와 같습니다.



$$
Var(X_2 \vert X_1) = E\bigg[\big\{X_2 - E(X_2 \vert X_1)\big\}^2 \big\vert X_1\bigg]
$$



확률 변수가 1개일 때, 분산을 다음과 $Var(X) = E(X^2) - E(X)^2$ 으로 나타낼 수 있었습니다. 이와 같이 조건부 확률의 분산도 식을 잘 정리하면 조금 더 간단하게 나타낼 수 있습니다.



$$
\begin{aligned}
E\bigg[\big\{X_2 - E(X_2 \vert X_1)\big\}^2 \vert X_1\bigg] &= E\bigg[{X_2}^2 - 2\cdot X_2 \cdot E(X_2 \vert X_1) + \big\{E(X_2 \vert X_1)\big\}^2 \vert X_1\bigg] \\
&= E[{X_2}^2 \vert X_1] - 2 \cdot E(X_2 \vert X_1) \cdot E[X_2 \vert X_1] + E(X_2 \vert X_1)^2 \\
&= E[{X_2}^2 \vert X_1] - E[X_2 \vert X_1]^2
\end{aligned}
$$



## Theorem

다음으로는 조건부 확률의 기댓값에 대한 몇 가지 정리를 알아보도록 하겠습니다. 첫 번째는 **이중 기댓값 정리(Double expectation theorem)**입니다. 이중 기댓값 정리를 사용하면 조건부로 주어진 확률 변수의 '기댓값의 기댓값'을 더욱 간단하게 나타낼 수 있습니다.



$$
E\bigg[E[X_2 \vert X_1]\bigg] = E(X_2)
$$



이중 기댓값 정리는 다음과 같이 증명할 수 있습니다.



$$
\begin{aligned}
E\bigg[E[X_2 \vert X_1]\bigg] &= \int \bigg[ \int x_2 \cdot f(x_2 \vert x_1) dx_2 \bigg] f(x_1)dx_1 \\
&= \int \int x_2 \cdot \frac{f(x_1,x_2)}{f(x_1)} \cdot f(x_1) dx_1dx_2 \\
&= \int x_2 \bigg(\int f(x_1,x_2) dx_1\bigg)dx_2 \\
&= \int x_2 f(x_2)dx_2 \\
&= E(X_2)
\end{aligned}
$$



다음으로 알아볼 정리는 단일 확률 변수의 분산을 조건부로 주어진 확률 변수의 분산과 기댓값으로 분리하는 정리입니다. 먼저 정리의 결과부터 알아보도록 하겠습니다.


$$
Var(X_2) = E\big[Var(X_2 \vert X_1)\big] + Var\big[E(X_2 \vert X_1)\big]
$$


수식의 길이만 보면 왼쪽의 길이가 더 짧고 간단해 보입니다. 하지만 이 정리를 사용하여 식을 변환했을 때 더욱 간단하게 분산값을 구할 수 있는 경우도 있습니다. 모든 $X_2$에 대한 값을 알기는 어렵지만 특정한 조건 $X_1$ 을 부여했을 때 그 값을 더 쉬워지는 것이지요.

**베이즈 정리(Bayes' Theorem)** 역시 식이 복잡해짐에도 값을 구하기 위해서 변환해주었습니다. 이처럼 단일 확률 변수에 대한 분산값을 구하기 위해서 특정한 조건을 부여하여 그 값을 구해내기도 합니다. 이제 분산에 대한 식으로부터 위 정리를 증명해보도록 하겠습니다.



$$
\begin{aligned}
Var(X_2) &= E\bigg[\{X_2 - E(X_2)\}^2\bigg] \\ 
&= E\bigg[\{X_2 - E(X_2 \vert X_1) + E(X_2 \vert X_1) - E(X_2)\}^2\bigg] \\
&= E\bigg[\{(X_2 - E(X_2 \vert X_1)) + (E(X_2 \vert X_1) - E(X_2))\}^2\bigg] \\
&= \color{red}{E\bigg[\{(X_2 - E(X_2 \vert X_1)\}^2\bigg]} + \color{blue}{E\bigg[\{E(X_2 \vert X_1) - E(X_2)\}^2\bigg]} \\
& \quad +2 \cdot \color{olive}{E\bigg[\{(X_2 - E(X_2 \vert X_1)\}\cdot\{(E(X_2 \vert X_1) - E(X_2))\}\bigg]}
\end{aligned}
$$



위 식에서 먼저 빨간색으로 나타낸 식을 간단하게 변환해보겠습니다. 변환 과정에 이중 기댓값 정리를 역방향으로 적용합니다.


$$
\begin{aligned}
\color{red}{E\bigg[\{(X_2 - E(X_2 \vert X_1)\}^2\bigg]} &= E\bigg[E\bigg[\{X_2 - E(X_2 \vert X_1)\}^2 \bigg\vert X_1 \bigg]\bigg]\\
& = E\bigg[Var(X_2\vert X_1)\bigg]
\end{aligned}
$$


다음으로는 파란색으로 나타낸 식을 간단하게 변환해보겠습니다. 아래 식의 변환 과정에도 역시 이중 기댓값 정리를 역방향으로 적용합니다.


$$
\begin{aligned}
\color{blue}{E\bigg[\{E(X_2 \vert X_1) - E(X_2)\}^2\bigg]} &= E\bigg[\{E(X_2 \vert X_1) - E[E(X_2 \vert X_1)]\}^2\bigg] \\
&= E\bigg[Var(X_2 \vert X_1)\bigg]
\end{aligned}
$$


마지막으로 올리브색으로 나타낸 부분을 변환하면 아래와 같습니다.


$$
\begin{aligned}
&\color{olive}{E\bigg[\{(X_2 - E(X_2 \vert X_1)\}\cdot\{(E(X_2 \vert X_1) - E(X_2))\}\bigg]} \\
=& E\bigg[(X_2 - E(X_2 \vert X_1)\bigg] \cdot \{E(X_2 \vert X_1) - E(X_2))\} \\
=& E\bigg[(X_2 - E(X_2 \vert X_1)\bigg] \cdot \{E(X_2 \vert X_1) - E(X_2))\} \\
=& \bigg[E(X_2) - E[E(X_2 \vert X_1)]\bigg] \cdot \{E(X_2 \vert X_1) - E(X_2))\}  \\
=& \bigg[E(X_2) - E(X_2)\bigg]\cdot \{E(X_2 \vert X_1) - E(X_2))\} \\
=& 0  \quad \because \text{Double Expectation Theorem}
\end{aligned}
$$



