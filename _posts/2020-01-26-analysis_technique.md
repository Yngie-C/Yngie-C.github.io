---
layout: post
title: Start :)
category: Daily
tag: Daily
---

 

본 포스트의 내용은 [권재명](https://dataninja.me/) 선생님의 [『따라하며 배우는 데이터 과학』](http://www.yes24.com/Product/Goods/44184320) 을 참조하여 작성하였습니다.

## 데이터에 맞는 분석을 하자

|             데이터형             |                      분석 기법과 R함수                       |                   모형                   |
| :------------------------------: | :----------------------------------------------------------: | :--------------------------------------: |
|          0. 모든 데이터          | 데이터 내용, 구조 파악(glimpse)<br>요약 통계량(summary)<br>단순 시각화(plot) |                                          |
|          1. 수량형 변수          | 분포시각(hist, boxplot, density)<br>요약 통계량(mean, median)<br>t-검정 t.test() |     $X_i \sim _iid N(\mu,\sigma^2)$      |
|    2. 범주형 변수(성공-실패)     | 도수 분포 table(), xtabs()<br>바 그래프 barplot()<br>이항검정 binom.test() |           $X \sim Binom(n,p)$            |
|      3. 수량형 x, 수량형 y       | 산점도 plot()<br>상관계수 cor()<br>단순회귀 lm()<br>로버스트 회귀 lqs()<br>비모수 회귀 | $(Y|X = x) \sim _iid N(\mu(x),\sigma^2)$ |
|      4. 범주형 x, 수량형 y       |     병렬 상자그림 boxplot()<br>분산분석(ANOVA) lm(y ~ x)     |   $Y_ij \sim _iid N(\mu_i, \sigma^2)$    |
| 5. 수량형 x, 범주형(성공-실패) y | 산점도/병렬상자그림 plot(), boxplot()<br>로지스틱 회귀분석 glm(family='binom') |      $Y = 1|X=x \sim Binom(1,p(x))$      |



