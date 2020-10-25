---
layout: post
title: 기댓값(Expect Value)
category: Statistics
tag: Statistics
---



이 게시물은 [부산대학교 김충락 교수님의 수리통계학 강의](http://kocw.net/home/search/kemView.do?kemId=1363783)를 참고하여 작성하였습니다.

# Expect Value

중학 수학에서 도수분포표로부터 값의 평균을 구해본 적이 있을 것입니다. 예를 들어 아래와 같은 도수분포표로부터 학생 키의 평균을 구하는 문제가 있다고 하겠습니다.

|  키(cm)   | 학생 수 | 상대도수 |
| :-------: | :-----: | :------: |
| 150 ~ 160 |    2    |   0.1    |
| 160 ~ 170 |    6    |   0.3    |
| 170 ~ 180 |    7    |   0.35   |
| 180 ~ 190 |    4    |   0.2    |
| 190 ~ 200 |    1    |   0.05   |

위 표에서 학생 키의 평균을 구하려면 각 계급의 계급값을 구한 후 이 값을 상대도수(=해당 계급의 학생 수 / 전체 학생 수)와 곱하면 되었습니다. 실제로 구해보면 아래와 같습니다.


$$
155 \times 0.1 + 165 \times 0.3 + 175 \times 0.35 + 185 \times 0.2 + 195 \times 0.05 = 173(cm)
$$


확률 변수의 기댓값을 구하는 방법도 위와 유사합니다. 확률 변수 $X$ 의 기댓값 $E(X)$는 각 확률 변수의 값 $x$와 그 때의 확률 밀도(질량) 함수의 값인 $f(x)$를 전부 더하여 사용합니다. 만약 $X$가 연속형 확률 변수라면 기댓값은 아래와 같이 각 값을 적분하여 구할 수 있습니다.


$$
E(X) = \int^\infty_{-\infty} xf(x)dx \qquad \text{if } \int^\infty_{-\infty} \vert x\vert f(x)dx < \infty
$$


$X$가 이산형 확률 변수라면 따로 떨어진 확률 변수에 대해서만 더해주어야 하므로 기댓값은 아래와 같이 구해지게 됩니다.


$$
E(X) = \sum^\infty_{-\infty} xf(x)dx \qquad \text{if } \sum^\infty_{-\infty} \vert x\vert f(x)dx < \infty
$$


구하고자 하는 기댓값의 대상이 변하게 되면 기댓값은 어떻게 변하게 될까요? 위 식들을 일반화하여 나타낸 함수 $g(x)$의 기댓값은 아래와 같이 나타낼 수 있습니다.


$$
E(g(x))
\begin{cases}
\int^\infty_{-\infty} g(x)f(x)dx \qquad \text{Continuous}\\
\\
\sum^\infty_{-\infty} g(x)f(x)dx \qquad \text{Discrete}
\end{cases}
$$



## (Population) Mean & Variance

기댓값을 이용하면 모평균(Population mean, $\mu$ )과 모분산(Population variance, $\sigma^2$)을 구할 수 있습니다. 먼저 모평균은 확률변수의 기댓값으로 정의되며 수식으로 나타내면 다음과 같습니다.


$$
\mu := E(X)
$$


모분산은 편차(각 인스턴스가 평균으로부터 얼마나 떨어져 있는지) 제곱의 합이므로 아래와 같이 나타낼 수 있습니다.


$$
\sigma^2 := E[(X-E(X))^2]
$$


위 식은 아래와 같이 변형할 수 있습니다.


$$
\begin{aligned}
E[(X-E(X)^2)] &= E[(X-\mu)^2] \\
&=\int(x-\mu)^2f_X(x)dx \\
&=\int x^2f_X(x)dx -2\int \mu \cdot xf_X(x)dx + \int \mu^2f_X(x)dx \\
&=\int x^2f_X(x)dx -2\mu\int xf_X(x)dx + \mu^2\int f_X(x)dx \\
&=\int x^2f_X(x)dx -2\mu^2 + \mu^2 \qquad (\because \int xf_X(x)dx=\mu, \int f_X(x)dx=1)\\
&=\int x^2f_X(x)dx -\mu^2 \\
&=E(X^2) - [E(X)]^2
\end{aligned}
$$


즉 분산은 확률변수 값 제곱의 기댓값에서 기댓값의 제곱을 빼주어 구할 수도 있게 됩니다. 이 때 확률 변수를 $k$ 제곱 한 것의 기댓값, $E(X^k)$ 를 $k$-차 모멘트(Moment)라고 합니다. 즉, 평균은 1차 모멘트이며 분산은 2차 모멘트에서 1차 모멘트의 제곱을 빼준 것이 됩니다. 모멘트에 대해서는 다음 게시물인 [적률 생성 함수(Moment Generating Function)](https://yngie-c.github.io/statistics/2020/10/23/mgf/)에서 더 자세히 논의를 이어나갈 것입니다.