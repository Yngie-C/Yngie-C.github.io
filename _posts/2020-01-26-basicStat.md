---
layout: post
title: 기초 통계(Basic statistics)
category: Statistics
tag: Statistics
---



본 포스트의 내용은 [권재명](https://dataninja.me/) 선생님의 [『따라하며 배우는 데이터 과학』](http://www.yes24.com/Product/Goods/44184320) 을  기반으로 작성하였습니다.

### 1. t-value

__t-value__를 알아보기 전에 __Z-value__를 먼저 알아보자.
$$
Z=\frac{\overline{X}-\mu_0}{\sqrt{\sigma^2/n}}
$$

$$
t
$$


### 2. P-value

학부생 때 들었던 통계학에서도 P-value가 특정 숫자 이상인지 아닌지 따졌을 뿐, 그 값의 의미 자체에 집중하지는 않았던 것 같다. P-value의 정의는 다음과 같다.

> 귀무가설 하에서 관찰된 통계량만큼의 극단적인 값이 관찰될 확률(The p-value is defined as the probability, under the assumption of the null hypothesis, of obtaining a result equal to or more extreme than what was actually observed)

예를 들어, P = 0.5라는 것은 귀무가설 하에서도 주어진 데이터만큼 크거나 작은 것이 충분히 관측될 만하다. 즉, 귀무가설에 대한 반박증거가 부족하다는 것이다. 반대로 P = 0.000001이라는 것은 귀무가설 하에서는 주어진 데이터만큼 크거나 작은 것이 관측될 확률이 거의 없다는 것이다.<br/>즉, P-value은 숫자일 뿐 모든 것을 결정해주지 않는다.

##### 미국통계학회의 P-value에 대한 설명서

1. P-value는 가정된 모형이 데이터와 별로 맞지 않음을 나타낼 수 있다.
2. P-value는 주어진 가설이 참일 확률이나, 데이터가 랜덤하게 생성된 확률이 아니다.
3. 과학적 연구 결과와 비즈니스, 정책 결정과정은 P-value가 어떤 경계값보다 크거나 작은 것에 근거해서는 안 된다.
4. 제대로 된 추론을 위해서는 연구과정 전반에 대한 보고서와 투명성이 필요하다.
5. P-value나 통계적 유의성은 효과의 크기나 결과의 중요성을 나타내지 않는다.
6. P-value 자체만으로는 모형이나 가설에 대한 증거가 되지 못한다.



### 3.  신뢰구간(Confidence Interval, CI)

신뢰구간의 정의도 살펴보자. 95% 신뢰구간의 정의는 다음과 같다.

> 같은 모형에서 반복해서 표본을 얻고, 신뢰구간을 얻을 때 신뢰구간이 참 모수값을 포함할 확률이 95%가 되도록 만들어진 구간(Were this procedure to be repeated on multiple samples, the calculated confidence interval would encompass the true population parameter 95% of the time)