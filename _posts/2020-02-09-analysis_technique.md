---
layout: post
title: 통계분석 기법들
category: R-language
tag: R
---

 

본 포스트의 내용은 [『따라하며 배우는 데이터 과학』](http://www.yes24.com/Product/Goods/44184320) , 권재명, 제이펍, 2017 을 참조하여 작성하였습니다.

<br/>

## 데이터에 맞는 분석을 하자 !

|             데이터형             |                      분석 기법과 R함수                       |                            모형                            |
| :------------------------------: | :----------------------------------------------------------: | :--------------------------------------------------------: |
|          0. 모든 데이터          | 데이터 내용, 구조 파악(glimpse)<br>요약 통계량(summary)<br>단순 시각화(plot) |                                                            |
|          1. 수량형 변수          | 분포시각(hist, boxplot, density)<br>요약 통계량(mean, median)<br>t-검정 t.test() |          $X_i \sim N_{\text{iid}}(\mu,\sigma^2)$           |
|    2. 범주형 변수(성공-실패)     | 도수 분포 table(), xtabs()<br>바 그래프 barplot()<br>이항검정 binom.test() |                 $X \sim \text{Binom}(n,p)$                 |
|      3. 수량형 x, 수량형 y       | 산점도 plot()<br>상관계수 cor()<br>단순회귀 lm()<br>로버스트 회귀 lqs()<br>비모수 회귀 | $(Y\vert\vert X = x) \sim N_{\text{iid}}(\mu(x),\sigma^2)$ |
|      4. 범주형 x, 수량형 y       |     병렬 상자그림 boxplot()<br>분산분석(ANOVA) lm(y ~ x)     |       $Y_{ij} \sim N_{\text{iid}}(\mu_i, \sigma^2)$        |
| 5. 수량형 x, 범주형(성공-실패) y | 산점도/병렬상자그림 plot(), boxplot()<br>로지스틱 회귀분석 glm(family='binom') |      $Y = 1 \vert\vert X=x \sim \text{Binom}(1,p(x))$      |



<br/>

## 0. 모든 데이터에 대하여

- `dplyr::glimpse()` 로 데이터를 훑어보자. 기본 패키지 함수에는 데이터 구조를 파악할 수 있는 `str()` , 데이터의 처음 부분을 보여주는 `head()` 등이 있다.
- `summary()` 로 요약 통계량과 결측치의 개수를 파악하자. 
- `plots(), pairs()` 로 무작정 시각화 해보자. 데이터의 개수가 많을 경우에는 실행시간이 오래 걸릴 수 있으므로 `dplyr::sample_n()` 을 활용하여 표본화한 후에 시도하는 것이 좋다. 



<br/>

## 1. 수량형 변수의 분석

- **시각화** (Visualiztion) : 히스토그램, 상자그림, 확률밀도함수 시각화를 통해 분포를 살펴본다. 베이스 패키지의 `hist(), boxplot()` 을 사용할 수 있다. 좀 더 추천되는 방법은 `ggplot() + geom_{histogram, density}()` 등의 함수를 사용하는 것이다.

- 데이터의 **정규성** 검사 : `qqplot()` , `qqline()` 를 통해 주어진 데이터가 정규분포와 얼마나 유사한지 검사한다.
- 가설검정과 신뢰구간 : `t.test()` 를 사용하여 **일변량 t-test** 를 수행할 수 있다.
  - 1변량 t-test : *일변량 연속형*  함수에 흔하게 사용되는 통계 추정 절차는 t-test이다. 다음과 같은 단측( *one-sided, "greater"* ) 검정을 시행하고자 한다고 해보자.

$$
H0: \text{mu} \leq 22.9 \quad \text{vs} \quad H1: \text{mu} \geq 22.9
$$

위의 귀무가설과 대립가설을 검증하기 위한 일변량 t-test 코드는 아래와 같다.

```R
# 책의 예시를 그대로 사용하겠습니다.
hwy <- mpg$hwy
mu0 <- 22.9
t.test(col_name, mu=mu0, alternative="greater")
>>>
data:  col_name
t = 1.3877, df = 233, p-value=0.08328
alternative hypothesis: true mean is greater than 22.9
95percent confidence inteval:
	22.79733	Inf
sample estimates:
mean of x
23.44017
```

*P-value* 해석하기 : *P-value* = 0.083 이다. 실제 고속도로 연비의 모평균이 22.9라면 우리가 관측한 것만큼 큰 표본평균값과 t 통계량(1.3877)이 관측될 확률은 8.3%라는 것이다. 따라서 유의수준 $\alpha$ 가 10%라면 고속도로 연비가 22.9보다 크다고 결론지을 수 있지만, 유의수준이 5%라면 고속도로 연비가 22.9보다 크다고 결론지을 만한 증거가 충분하지 않다고 할 수 있다.

*신뢰구간* 표기 : 위 결과에서 95% 신뢰구간은 약 [22.8 , ∞ ) 이며, 같은 데이터로 양측( *two_sided* ) 검정을 했을 때의 신뢰구간은 약 [22.7, 24.2] 임을 알 수 있다.

`alternative = "two-sided"` 로 설정하면 양측 검정을 할 수 있으며,  95% 가 아닌 다른 신뢰도에서 t-검정을 하고 싶다면 `conf.level=` 옵션을 사용하여 조정할 수 있다. 

- 이상점 찾아보기 : 로버스트 통계량 계산

(일단 생략)

<br/>

## 2. 성공-실패(1-0)값 범주형 변수의 분석

- 요약 통계량 계산 : `table(), xtabs()` 등이 있다. `prop.table()` 함수는 도수를 상대도수로 바꿔준다. 이들 중 `table()` 함수는 도수 분포를 계산해주고, `xtabs()` 함수는 도수 분포 계산뿐만 아니라 formula 인터페이스도 사용할 수 있다. `prop.table()` 함수를 이용하면 상대도수를 구할 수 있다.
- 시각화 : `boxplot()` 이 유용하다.
- 가설검정과 신뢰구간 : `binom.test()` 함수를 사용하면 '성공률'에 대한 검정과 신뢰구간을 구할 수 있다.

<br/>

## 3. 독립변수와 종속변수

(책에서는 '설명변수와 반응변수'라는 표현을 사용하였으나 독립변수와 종속변수로 표기하였습니다.)

두 변수 사이의 관계를 연구할 때는 각 변수를 독립변수(Independent Variable, $X$ )와 종속변수(Dependent Variable, $Y$ )로 구분하는 것이 유용하다. 보통 인과 관계에서 원인이 되는 것으로 믿어지는 변수를 $X$ 로 결과가 되는 변수를 $Y$ 로 놓는다.

<br/>

## 4. 수량형 $X$ , 수량형 $Y$ 의 분석

- **산점도** : 산점도를 통해 관계의 모양을 파악한다. 베이스 패키지의 `plot()` 이나 `ggplot2()` 의 `geom_point()`를 사용한다. 중복치가 많을 때에는 `jitter` 를 사용하며, 데이터 수가 너무 많을 때는 `alpha=` 옵션을 사용하거나 표본화한다. 관계가 선형인지, 강한지 약한지, 이상치는 있는지 등을 파악한다. 이를 통해 이상점의 유무 등을 파악해낸다. 아래는 산점도를 그리기 위한 R 코드이다.

```R
ggplot(data, aes(X, Y)) + geom_jitter() + geom_smooth(method='lm')
# X, Y는 각각 X축, Y축에 속하는 특성의 이름이다.
```

- **상관계수** : `cor()` 함수를 통해 상관계수를 계산한다. 기본적으로 피어슨(Pearson) 상관계수를 계산해준다. 피어슨 상관계수 아래의 식을 통해 구해진다.

$$
r = \frac{\sum^n_{i=1} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum^n_{i=1}(x_i - \bar{x})^2} \sqrt{\sum^n_{i=1}(y_i - \bar{y})^2}}
$$

피어슨 상관계수는 두 변량의 '선형(linear)' 관계의 강도를 -1에서 1사이의 숫자로 나타낸다. 0은 상관관계가 없음을 나타낸다. 피어슨 상관관계는 선형 강도의 관계만 나타내 줄 뿐, 관계의 기울기를 이야기해 주지는 않는다. 따라서 상관관계를 구하기 이전에 산점도를 통해 관계의 기울기, 혹은 데이터의 군집성 등을 파악해주는 과정이 필요하다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Correlation_examples2.svg/600px-Correlation_examples2.svg.png" alt="상관계수" style="zoom:100%;" /></p>
또한, 상관계수는 이상치의 영향을 많이 받기 때문에 로버스트한 방법인 스피어만(Spearman) 상관계수나 켄달(Kendall)의 $\tau$ (타우)를 사용하여 통계량을 계산하기도 한다.

- 켄달의 타우( $\tau$ )

$$
\tau = \frac{\text{# of concordant data} - \text{# of discordant data}}{n(n-1)/2}
$$

$\text{concordant}$ 는 ( $x_i > x_j$ 이고 $y_i > y_j$) 또는 ( $x_i < x_j$ 이고 $y_i < y_j$ ) 인 데이터를 의미하며, $\text{discordant}$ 는 ($x_i > x_j$ 이고 $y_i < y_j$ ) 또는 ( $x_i < x_j$ 이고 $y_i > y_j$ )인 데이터를 의미한다.

스피어만의 상관계수는 순위상관계수(rank correlation coefficient)라고 불린다. 각 $x_i$ 와 $y_i$ 를 순위값(1, 2, 3, ... , n)으로 바꿔준 후, 피어슨 상관계수를 계산해준 값이다. 원래의 $x_i$ 와 $y_i$ 의 값이 극단적이라도 1~n 사이의 값으로 제한되어 이상치의 영향을 덜 받게 된다. 아래는 피어슨 상관계수와 스피어만 상관계수를 비교한 그래프이다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Spearman_fig3.svg/1024px-Spearman_fig3.svg.png" alt="스피어만2" style="zoom:33%;" /></p>
- **선형회귀** 모형 적합 : 여러 독립변수를 사용하여 수량형 종속변수 $Y$ 를 예측하기 위한 유용한 방법은 선형회귀 모델이다. 

$$
Y \sim \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... \beta_px_{ip} + \epsilon_i \quad \epsilon_i\sim N_{\text{iid}}(0,\sigma^2)
$$

$x_{ij}$ 는 $j$ 번째 설명변수의 $i$ 번째 관측치를 나타내며,  $_{\text{iid}}$ 는  독립이고 동일한 분포를 따름[^1]을 나타낸다. $\beta_0$ 는 절편(Intercept), $(\beta_1, \beta_2, \beta_p)$ 는 기울기(Slope)를 나타내며 $\epsilon_i$ 는 오차항이다. 이처럼 하나의 $X$ 변수로 이루어진 모형을 단순 회귀분석 모형이라고 한다. `lm()` 과 `summary.lm()` 잔차의 제곱합을 최소화하는 최소제곱법(Least Squares Method)으로 선형 모형을 추정한다. 아래는 선형회귀를 위한 R코드이다.

```R
data1_lm <- lm(Y ~ X, data=data1)
summary(data1_lm)
```

- 모형 **적합도 검정** : `summary()` 를 통해 출력된 마지막 두 줄은 모형 전체의 설명력을 나타낸다. 마지막 두 줄의 첫번째 결과인 Multiple R-squared $(R^2)$ 는 결정계수(Coefficient of Determination)라고 불린다. 종속변수의 총 변동 중 얼마만큼이 선형 모형으로 설명이 되는지 나타낸다. [0,1] 범위의 값을 나타내며 1에 가까워질 수록 설명력이 높음을 나타낸다. 아래는 $R^2$ 를 위한 공식이다.

$$
R^2 = \frac{SSR}{SST} = \frac{SSR}{SSR+SSE} = \frac{\text{회귀분석으로 설명되는 종속변수의 변동}}{\text{모형화 전 종속변수의 변동}}
$$

여기서 *총 제곱합* $SST = \sum^n_{i=1}(y_i - \bar{y})$ 는 모형화 전의 종속변수의 변동을 나타낸다. 총 제곱합은 회귀 제곱합과 잔차 제곱합으로 나뉜다. *회귀 제곱합* $SSR = \sum^n_{i=1}(\hat{y_i} - \bar{y})$ 은 회귀분석으로 설명되는 종속변수의 변동을 나타내며, *잔차 제곱합* $SSE = \sum^n_{i=1}(y_i - \hat{y})$ 는 모형으로 설명되지 않는 반응변수의 변동을 나타낸다.

- $\text{Multiple }R^2$ 값을 볼 때 주의해야 할 점이 있다. 회귀모형에서는 독립변수의 개수를 추가할수록 $R^2$ 값이 항상 증가한다는 것이다. 즉, 아무리 상관없는 독립변수를 모형에 추가하더라도 모형의 설명력은 올라가게 되어 있다. $\text{Adjust }R^2$ 은 이런 문제를 보완하기 위한 값이다. 아래는 $\text{Adjust }R^2$ 을 구하기위한 공식이다. $p$ 는 $X$ 변수의 개수이며 $p$ 가 커질수록 적당한 보정을 하여 무작정 특성을 추가하면 $R^2$ 가 낮아지도록 해준다.

$$
\text{Adjust }R^2 = 1-(1-R^2)\frac{n-1}{n-p-1} = R^2-(1-R^2)\frac{p}{n-p-1}
$$

- **선형회귀 모형 예측** : R의 `predict()` 함수는 종속변수의 예측값 $\hat{y_i} = \hat{\beta_0} + \hat{\beta_1}x_i$ 을 계산한다. `resid()` 함수는 잔차 $\epsilon_i = y_i - \hat{y_i}$ 를 계산해준다.
- 선형회귀 모델의 가정 진단 : 선형회귀 모델이 의미있기 위해서는 다음의 여러 조건을 충족하여야 한다.

1. 독립변수와 종속변수의 관계가 선형이다.
2. 잔차의 분포가 독립이다.
3. 잔차의 분포가 동일하다.
4. 잔차의 분포가 $N(0, \sigma^2)$ 이다.

- 로버스트 선형 회귀분석

- 비선형/비모수적 방법, 평활법과 LOESS

<br/>

## 5. 범주형 $X$, 수량형 $Y$

- **분산분석** : 독립변수가 범주형이고 종속변수가 수량형일 경우에는 선형 모형의 특별한 예인 분산분석(Analysis of Variance, **ANOVA** )을 사용한다. (집단의 개수가 2개일 경우에 사용하는 2-sample t-test는 특별한 경우임) 수학적으로는 원소가 0과 1로 이루어진 $n \times p$ 행렬을 사용하여 아래와 같이 표현할 수 있다. 

$$
Y_{i} \sim \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... \beta_px_{ip} + \epsilon_i \quad \epsilon_i\sim N_{\text{iid}}(0,\sigma^2)
$$

$i = 1, 2, ... , p$ 는 범주변수 $x$ 의 값에 따라 데이터를 나눈 서브셋이며, $j = 1, 2, ... ,n_i$ 는 각 그룹에 있는 관측치의 개수이다.

- **선형모형과 t-test** : 앞서 살펴본 회귀분석이나 지금 살펴보는 ANOVA 분석이나 수학적으로는 동일한 선형모형이다. 분산분석 또한 R의 `lm()` 함수를 통해 적용할 수 있다.

```R
data1_lm2 <- lm(Y ~ Xc, data=data1)
summary(data1_lm2)
```

선형모형과 마찬가지로 `summary()` 로 알아본 결과의 마지막 두 줄은 분산분석 모형의 적합도를 나타낸다.  $\text{Multiple }R^2$ 는 해당 범주형 독립변수로 설명되는 종속변수의 비율이다. $\text{Adjust }R^2$ 는 모형의 복잡도(범주형 변수 내의 레이블 개수)를 감안한 $R^2$ 값이다.

- 분산분석의 진단 : 분산분석이 의미있기 위해서는 다음 가정이 충족되어야 한다.

1. 잔차의 분포가 독립이다.
2. 잔차의 분산이 동일하다.
3. 잔차의 분포가 $N(0, \sigma^2)$ 이다.



<br/>

## 6) 수량형 $X$, 범주형 $Y$ (0-1)

- 일반화 선형 모형, 로짓/로지스틱 함수 : 0-1 형태의 범주형 독립변수와 수량형(혹은 범주형) 독립변수를 가진 데이터는 전통적인 선형모형으로 다룰 수 없다. 대신 일반화 선형모형(Generalized Linear Model, GLM), 특히 이항분포 패밀리(binomial family)를 사용하여야 한다. 독립변수는 0과 1의 값을 가지는 베르누이 확률변수이다. $Y$ 가 1일 확률은 0과 1사이의 값으로, 독립변수 $x$ 의 함수 형태인 $\mu(x)$ 로 나타낸다.

$$
Y \sim \text{Bernoulli(p)} \quad → \quad \text{Pr}(Y=1\vert x) = \mu(x)
$$

확률값 $\mu(x)$ 는 선형예측 함수인 $x\beta$ 와 로짓(Logit)함수로 연결되어 있다.

- **로짓 함수**

$$
\text{logit}(\mu) = \log{(\frac{\mu}{1-\mu})} = \eta(x) = x\beta
$$

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Logit.svg/525px-Logit.svg.png" alt="로짓" style="zoom:50%;" /></p>
모수벡터 $\beta$  의 추정값 $\hat{\beta}$ 는 최대우도법(Maximum Likelihood Estimation, MLE)으로 계산한다. 최대우도법은 해를 구하는 공식이 존재하지 않으므로 Newton-Rhapson법을 사용하여 반복적으로 찾아나간다. 이를 통해 $\hat{\beta}$ 를 얻게되면, 종속변수의 기대값을 추정할 수 있다. 선형추정 값인 $x\hat{\beta}$ 가 무한한 값을 가지므로 이를 (0,1) 범위의 확률값으로 변환하기 위해 로지스틱(Logistic) 함수를 사용한다.

- **로지스틱 함수**

$$
\hat{\mu}(x) = \text{logit}^{-1}(\hat{\eta}) = \text{logistic}(\hat{\eta}) = \frac{1}{1 + \exp{(-\hat{\eta}})} = \frac{\exp({\hat{\eta}})}{\exp({\hat{\eta}}) +1}
$$



<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1280px-Logistic-curve.svg.png" alt="로지스틱" style="zoom:25%;" /></p>
로지스틱 회귀 모델을 적용하기 위한 R 코드는 다음과 같다.

```R
data1_glm <- glm(Y ~ X, data=data1, family='binomial')
summary(chall_glm)
```

- GLM 모형 적합도 : glm 또한 `summary()` 를 통한 출력결과 중 마지막 2~4줄로 모델의 적합도를 판단할 수 있다.
- 로지스틱 모형 예측, 링크와 반응변수
- 로지스틱 모형 적합결과의 시각화
- 범주형 종속변수 $Y$  범주가 셋 이상일 경우
- GLM 모형의 일반화

[^1]: Independent and Identically distributed 의 약자이다.