---
layout: post
title: 분석 기법들
category: R-language
tag: R
---

 

본 포스트의 내용은 [권재명](https://dataninja.me/) 선생님의 [『따라하며 배우는 데이터 과학』](http://www.yes24.com/Product/Goods/44184320) 을 참조하여 작성하였습니다.

<br/>

## 데이터에 맞는 분석을 하자 !

|             데이터형             |                      분석 기법과 R함수                       |                   모형                   |
| :------------------------------: | :----------------------------------------------------------: | :--------------------------------------: |
|          0. 모든 데이터          | 데이터 내용, 구조 파악(glimpse)<br>요약 통계량(summary)<br>단순 시각화(plot) |                                          |
|          1. 수량형 변수          | 분포시각(hist, boxplot, density)<br>요약 통계량(mean, median)<br>t-검정 t.test() |     $X_i \sim _iid N(\mu,\sigma^2)$      |
|    2. 범주형 변수(성공-실패)     | 도수 분포 table(), xtabs()<br>바 그래프 barplot()<br>이항검정 binom.test() |           $X \sim Binom(n,p)$            |
|      3. 수량형 x, 수량형 y       | 산점도 plot()<br>상관계수 cor()<br>단순회귀 lm()<br>로버스트 회귀 lqs()<br>비모수 회귀 | $(Y|X = x) \sim _iid N(\mu(x),\sigma^2)$ |
|      4. 범주형 x, 수량형 y       |     병렬 상자그림 boxplot()<br>분산분석(ANOVA) lm(y ~ x)     |   $Y_ij \sim _iid N(\mu_i, \sigma^2)$    |
| 5. 수량형 x, 범주형(성공-실패) y | 산점도/병렬상자그림 plot(), boxplot()<br>로지스틱 회귀분석 glm(family='binom') |      $Y = 1|X=x \sim Binom(1,p(x))$      |



<br/>

## 0. 모든 데이터에 대하여

- _dplyr::glimpse()_ 로 데이터를 훑어보기
- _summary()_ 로 요약 통계량 및 결측치 파악하기
- _plots(), pairs()_ 로 무작정 시각화 해보기



<br/>

## 1. 수량형 변수의 분석

- 시각화 : 히스토그램, 상자그림, 확률밀도함수 시각화
  - 데이터 시각화
- 데이터의 정규성 검사 : _qqplot_ , _qqline()_ 를 통해 주어진 데이터가 정규분포와 얼마나 유사한지 검사
- 가설검정과 신뢰구간 : _t.test()_ 를 사용하여 1변량 t-test, 신뢰구간을 구할 수 있다.
- 이상점 찾아보기 : 로버스트 통계량 계산



<br/>

## 2. 성공-실패(1-0)값 범주형 변수의 분석