---
layout: post
title: 기초 통계(Basic statistics)
category: R-language
tag: Statistics
---



본 포스트의 내용은 [『따라하며 배우는 데이터 과학』](http://www.yes24.com/Product/Goods/44184320) , 권재명, 제이펍, 2017 을  기반으로 작성하였습니다.



## 1. t-value

- **Z-value**  

$$
Z=\frac{\overline{X}-\mu_0}{\sqrt{\sigma^2/n}} \sim N(0,1) \qquad \text{When} \quad H_0 : \mu = \mu_0
$$

*Z-value* 의 문제점은 모분산 ${\sigma^2}$ 을 알 수 없다는 것이다.

- **t-value** : 모분산 (${\sigma^2}$ )을 표본분산($s^2$)으로 추정한 통계량.

$$
t=\frac{\overline{X}-\mu_0}{\sqrt{s^2/n}} \sim t_v \qquad \text{When} \quad H_0 : \mu = \mu_0
$$

$v = n-1$ (자유도, Degree of Freedom, DoE)가 클수록 $N(0,1)$에 가까워진다.

<br/>

## 2. P-value

P-value의 정의는 다음과 같다.

> 귀무가설 하에서 관찰된 통계량만큼의 극단적인 값이 관찰될 확률(The p-value is defined as the probability, under the assumption of the null hypothesis, of obtaining a result equal to or more extreme than what was actually observed)

예를 들어, P = 0.5라는 것은 귀무가설 하에서도 주어진 데이터만큼 크거나 작은 것이 충분히 관측될 만하다. 즉, 귀무가설에 대한 반박증거가 부족하다는 것이다. 반대로 P = 0.000001이라는 것은 귀무가설 하에서는 주어진 데이터만큼 크거나 작은 것이 관측될 확률이 거의 없다는 것이다.

즉, P-value은 숫자일 뿐 모든 것을 결정해주지 않는다.

**미국통계학회의 P-value에 대한 설명서**

1. P-value는 가정된 모형이 데이터와 별로 맞지 않음을 나타낼 수 있다.
2. P-value는 주어진 가설이 참일 확률이나, 데이터가 랜덤하게 생성된 확률이 아니다.
3. 과학적 연구 결과와 비즈니스, 정책 결정과정은 P-value가 어떤 경계값보다 크거나 작은 것에 근거해서는 안 된다.
4. 제대로 된 추론을 위해서는 연구과정 전반에 대한 보고서와 투명성이 필요하다.
5. P-value나 통계적 유의성은 효과의 크기나 결과의 중요성을 나타내지 않는다.
6. P-value 자체만으로는 모형이나 가설에 대한 증거가 되지 못한다.



<br/>

## 3. 신뢰구간(Confidence Interval, CI)

신뢰구간의 정의도 살펴보자. 95% 신뢰구간의 정의는 다음과 같다.

> 같은 모형에서 반복해서 표본을 얻고, 신뢰구간을 얻을 때 신뢰구간이 참 모수값을 포함할 확률이 95%가 되도록 만들어진 구간(Were this procedure to be repeated on multiple samples, the calculated confidence interval would encompass the true population parameter 95% of the time)

쉽게 설명하자면, 95% 신뢰구간을 이용하여 100편의 논문을 사용했다면 그 중 5개 논문의 결론은 잘못되었다고 할 수 있다.



<br/>

## 4. 이외의 다른 용어들

- __모집단(population)__ : 데이터가 표본화되었다고 가정하는 분포/집단
- __모수(population parameter)__ : 모집단을 정의하는 미지의 상수
- __표본(sample)__ : 모집단으로부터 랜덤하게 추출된 일부 관측치
- __통계량(statistics)__ : 모수를 추정하기 위해 데이터로부터 계산된 값
- __귀무가설(null hypothesis)__ : 모수에 대한 기존(status quo)의 사실 혹은 디폴트 값
- __대립가설(alternative hypothesis)__ : 모수에 대해 귀무가설과 대립하여 증명하고 싶은 사실
- __가설검정(hypothesis testing)__ : 통계량을 사용해 귀무가설을 기각하는 절차
- __타입 1 오류(Type 1 error)__ : 가설검정 절차가 참인 귀무가설을 기각하는 사건
- __타입 2 오류(Type 2 error)__ : 가설검정 절차가 거짓인 귀무가설을 기각하지 않는 사건
- __유의수준(significance level)__ : 타입 1 오류를 범할 확률의 허용치
- __P-값__ : 만약 귀무가설이 참일 때 데이터가 보여준 정도로 특이한 값이 관측될 확률
- __중심극한정리__ : 표본 평균이 항상 대략적으로 종모양(정규분포)을 따르는 것. 이 때문에 비교적 적은 수의 모형을 가지고 있다고 하더라도 분석을 위해 사용할 수 있다. 아래는 정규분포식.

$$
f(x) = \frac{1}{\sqrt{2\sigma^2\pi}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$



