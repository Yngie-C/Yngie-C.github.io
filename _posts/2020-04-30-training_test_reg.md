---
layout: post
title: 과적합과 과소적합 (Overfitting & Underfitting)
category: Machine Learning
tag: Machine-Learning
---

 본 포스트는  [문일철 교수님의 인공지능 및 기계학습 개론 I](https://www.edwith.org/machinelearning1_17/joinLectures/9738) 강의를 바탕으로 작성하였습니다.



# Overfitting & Underfitting

## Train Data and Test Data

머신러닝의 목적은 기계에게 데이터를 학습시켜 더 정확한 모델을 만들어 내도록 하는 것입니다. 기계를 학습시켜 모델을 만드는 과정을 수험생에 빗대어 생각해봅시다. 이 수험생은 학습 과정에서 교과서나 문제지에 있는 수많은 문제를 풀 것입니다. 학습 과정에서 익히게 되는 이런 문제가 머신러닝의 관점에서는 **학습 데이터셋(Train dataset)**이 됩니다. 교과서나 문제지를 기반으로 문제 유형을 파악하고 그에 맞게 풀이 방법을 익히는 것처럼 기계도 학습 데이터셋을 통해서 데이터에 맞는 모델을 생성하고 수정해나갑니다.

하지만 문제지를 잘 푸는 학생이라고 항상 좋은 등급을 줄 수는 없습니다. 그 학생이 어떤 방식으로 공부를 했는지(예를 들어, 문제지 답만 외우지는 않았는지)를 테스트하기 위한 수능같은 과정이 필요합니다. 문제지에 있는 문제만 잘 푸는지 아니면 전반적으로 실력이 좋은 학생인지 가늠하기 위한 단계이지요. 이렇게 테스트하기 위한 시험이 머신러닝 관점에서는 **시험 데이터셋(Test dataset)**이 됩니다. 모델도 실제 데이터에 적용하기 전에 훈련 데이터셋에만 잘 맞는지 아닌지를 가늠하기 위해 시험을 보아야 하는 것이지요.

중요한 점은 학습 과정 중에 시험 데이터셋을 기계가 보게 해서는 안된다는 것입니다. 수능 문제가 시중 교과서나 문제지에 없는 문제로 출시되는 이유와도 비슷합니다. 만약 시중 문제지에 있는 문제가 수능에 많이 나오게 된다면 수험생의 진정한 실력을 측정하는 데에 오차를 줄 수도 있습니다. 문제를 외워서 풀거나 하는 학생들이 있을 테니까요. (EBS 연계는 배제하고 생각하겠습니다 ㅠㅠ)

마찬가지로 시험 데이터셋이 학습 데이터셋의 일부로 들어가게 된다면, 시험 데이터셋은 이를 통해 학습한 기계를 제대로 평가하는 척도가 되지 못할 것입니다. 그렇기 때문에 실제로 머신러닝 과정에서는 학습 데이터셋과 시험 데이터셋을 철저히 분리해줍니다. 일반적으로 가지고 있는 전체 데이터셋 중 임의로 선택한 15~20% 정도를 시험 데이터셋으로 분류한 후 나머지 학습 데이터셋만으로 모델 학습을 진행합니다.

## Overfitting & Underfitting

모델을 생성하는 방법에 따라서 다양한 모델이 생성될 수 있습니다. 아래 그림을 봅시다. 이 그림은 주어진 데이터에 대해 다항 선형 회귀(Polynomial linear regression)를 사용하여 생성한 모델의 그래프를 각각 나타낸 것입니다.

<p align="center"><img src="https://www.educative.io/api/edpresso/shot/6668977167138816/image/5033807687188480" alt="over_under_fit" style="zoom: 100%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.educative.io/edpresso/overfitting-and-underfitting">educative.io</a></p>

먼저 가운데 그려진 그래프부터 살펴보겠습니다. 이 그래프를 그린 모델은 전체 변수 범위에서 데이터의 모양을 그럴듯하게 근사하고 있음을 알 수 있습니다. 물론 학습 데이터에 대해 100%의 적합도를 보여주지는 않지만 새로운 데이터가 들어오더라도 어느 정도 잘 맞을 것으로 예측할 수 있습니다.

왼쪽에 있는 선형 모델은 변수가 0인 쪽의 데이터 몇 개에 대해서는 비교적 잘 근사하는 듯합니다. 하지만, 나머지 데이터는 회귀 그래프에서 떨어져 있고, 변수가 큰 부분의 데이터는 우하향하고 있는데 회귀 그래프는 그대로 우상향하고 있는 것을 보면 제대로 회귀하지 못하고 있다고 볼 수 있습니다.

오른쪽에 그려진 그래프는 어떨까요? 학습 데이터와 생성된 모델의 오차를 구해보면 0에 가까울 것입니다. 회귀 그래프가 모든 점을 지나고 있는 것을 볼 수 있습니다. 가운데 그래프보다 훈련 데이터를 잘 만족하는 오른쪽 그래프를 만드는 모델의 성능이 더 좋은 것 아닐까요?

어떤 모델이 좋은 지를 가늠하기 위해서 시험 데이터 하나를 추가하여 모델과의 오차가 얼마인지를 비교해 보겠습니다. 시험 데이터는 아래 그림에서 빨간색 점으로 나타납니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/90508764-3c84ce00-e193-11ea-8350-9307399adaa4.png" alt="with_test_data"  /></p>

추가된 시험 데이터는 학습 데이터셋의 경향을 크게 벗어나지 않는 것처럼 보입니다. 그런데 생성된 각 모델과의 오차는 어떻게 될까요? 왼쪽 그래프부터 보면 오차가 상당히 많이 나는 것을 볼 수 있습니다. 훈련 데이터셋을 제대로 회귀하지 못했으니 그와 비슷한 시험 데이터를 비교했을 때에도 큰 오차를 보이는 것을 알 수 있습니다. 다음으로 가운데 모델을 보겠습니다. 가운데 모델은 오차가 크지 않습니다. 학습 데이터를 어느 정도 잘 근사했으니 시험 데이터도 비슷하게 근사하는 것을 볼 수 있습니다. 그렇다면 학습 데이터를 100%에 가깝게 근사했던 오른쪽 모델은 어떻게 될까요? 이 모델은 학습 데이터와의 오차가 없었음에도 시험 데이터와는 큰 오차를 보이는 것을 알 수 있습니다.

위 이미지에서 왼쪽 그래프와 같이 너무 단순한 모델을 생성하여 학습 데이터와 잘 맞지 않을 때 모델이 **과소적합(Underfitting)**되었다고 합니다. 이럴 때는 모델의 복잡도를 늘려줌으로써 문제를 해결해야 합니다. 반대로 오른쪽처럼 너무 복잡한 모델을 생성하는 바람에 학습 데이터에는 굉장히 잘 맞지만 새로운 데이터에는 잘 맞지 않는 현상을 **과적합(Overfitting, 과대적합)**이라고 합니다.

위에서 살펴본 바와 같이 왼쪽 그래프와 오른쪽 그래프는 시험 데이터와의 오차가 발생하는 이유가 다릅니다. 전자는 훈련 데이터도 제대로 근사를 못할 만큼 모델이 단순한 것이 문제였습니다. 이런 경우에 발생하는 오차를 **편향(Bias)**이라고 합니다. 후자는 훈련 데이터에 대해서는 근사를 매우 잘하지만 새로운 데이터가 들어왔을 때 제대로 근사하지 못한다는 문제가 있었습니다. 이런 경우에 발생하는 오차는 **분산(Variance)**이라고 합니다.

## Bias-Variance Tradeoff

단순한 모델에서 발생하는 편향을 줄이기 위해 모델의 복잡도를 늘리면 분산이 늘어나고, 복잡한 모델에서 분산을 줄이기 위해서 모델을 단순화 하면 편향이 늘어나게 됩니다. 편향과 분산은 한쪽을 줄이면 다른 한쪽이 증가하는 성질이 있습니다. 이를 **편향-분산 트레이드오프(Bias-Variance Tradeoff)**라고 합니다.

편향-분산 트레이드 오프가 발생하는 과정을 수식을 통해서 알아봅시다. 일단 우리가 생성한 모델에서 발생하는 모든 오차의 합을 $E_{\text{out}}$ 이라고 합시다. 이 오차를 위에서 살펴본 것처럼 두 가지로 나눠 생각해볼 수 있습니다.

하나는 알고리즘을 학습하는 과정(Approximation)에서 발생하는 오차 $E_{\text{in}}$ 입니다. 이 경우는 위에서 살펴본 편향에 해당합니다. 나머지 하나는 모든 데이터를 관측하지 못하는 데(Generalization)서 발생하는 오차 $\Omega$ 입니다. 이 오차는 위에서 살펴본 분산에 해당합니다. 이를 수식으로 나타내면 다음과 같이 나타낼 수 있습니다. 


$$
E_{\text{out}} \leq E_{\text{in}} + \Omega
$$

우리가 찾고자 하는 목적 함수를 $f$ 라 하고, 기계가 학습하여 생성한 모델의 함수를 $g$ 라고 합시다. 존재하는 데이터 $D$ 에 대해서 생성되는 모델의 함수를 $g^D$ 라고 해봅시다. 이를 사용하면 데이터셋 $D$ 안에 있는 인스턴스 하나에 의해 발생하는 오차를 다음과 같이 나타낼 수 있습니다.



$$
E_{\text{out}}(g^D(x)) = E_X \bigg[\big(g^D(x)-f(x)\big)^2\bigg]
$$



이를 데이터셋 전체에 대한 오차 기댓값을 계산하는 수식으로 변형할 수 있습니다.



$$
E_D[E_{\text{out}}(g^D(x)] = E_D\bigg[E_X \big[\big(g^D(x)-f(x)\big)^2\big] \bigg] = E_X\bigg[E_D \big[\big(g^D(x)-f(x)\big)^2\big] \bigg]
$$



약간의 트릭을 사용하면 $E_D \big[\big(g^D(x)-f(x)\big)^2\big]$ 부분을 간단하게 만들어줄 수 있습니다. 식 변형 과정 중에 등장하는 $\bar{g}(x)$ 는 무한 개의 데이터셋이 주어질 때 생성되는 모델의 평균으로 수식으로 나타내면 $\bar{g}(x) = E_D(g^D(x))$ 입니다.


$$
\begin{aligned}
E_D \big[\big(g^D(x)-f(x)\big)^2\big] &= E_D \big[\big(g^D(x)-\bar{g}(x) + \bar{g}(x) -f(x)\big)^2\big] \\
&= E_D \big[\big(g^D(x)-\bar{g}(x)\big)^2 + \big(\bar{g}(x) -f(x)\big)^2 + 2\big(g^D(x)-\bar{g}(x)\big)\big(\bar{g}(x) -f(x)\big)\big] \\
&= E_D \big[\big(g^D(x)-\bar{g}(x)\big)^2\big] + \big(\bar{g}(x) -f(x)\big)^2 + E_D\big[2\big(g^D(x)-\bar{g}(x)\big)\big(\bar{g}(x) -f(x)\big)\big] \\
&= E_D \big[\big(g^D(x)-\bar{g}(x)\big)^2\big] + \big(\bar{g}(x) -f(x)\big)^2\\
&\because \bar{g}(x) = E_D(g^D(x))
\end{aligned}
$$


변형한 식을 원래의 식에 대입하면 데이터 셋을 통해 생성된 모델로부터 발생하는 오차를 다음과 같은 수식으로 나타낼 수 있습니다.



$$
E_D[E_{\text{out}}(g^D(x)] = E_X\bigg[E_D \big[\big(g^D(x)-\bar{g}(x)\big)^2\big] + \big(\bar{g}(x) -f(x)\big)^2\bigg]
$$


유도된 식에서 $E_X$ 이하의 부분을 첫 번째 항과 두 번째 항으로 나눌 수 있습니다. 첫 번째 항은 **분산(Variance)**에 해당하는 항입니다. 수식 $E_D \big[\big(g^D(x)-\bar{g}(x)\big)^2\big]$ 으로부터 *'무한 개의 데이터셋으로부터 이끌어 낸 가설과 특정 데이터셋 $D$ 로부터 이끌어낸 가설 사이의 차이'* 라는 의미를 이끌어낼 수 있습니다. 이는 앞서 알아본 학습 데이터셋 이외의 데이터로부터 발생하는 오차의 의미이기도 하지요.

두 번째 항은 **편향(Bias)**의 제곱과 관련되어 있습니다. 분산의 수식으로부터 의미를 이끌어내 봅시다. $\big(\bar{g}(x) -f(x)\big)$ 로부터 *'무한 개의 데이터셋으로부터 이끌어 낸 가설과 목적 함수의 차이'* 라는 의미를 이끌어낼 수 있습니다. 이는 위에서 모델을 근사(approximation)하는 과정에서 학습의 한계 때문에 발생하는 오차라고 말한 것과 같습니다.

그렇다면 편향이 큰 모델과, 분산이 큰 모델 중 어떤 모델을 선택하는 것이 좋을까요? 특정 데이터셋을 통해 학습시킨 결과, 전체 오차 $E_{\text{out}}$는 같지만 편향과 분산은 다른 두 모델이 생성되었다고 합시다. 예를 들어, 한 모델은 편향과 분산의 비율이 $7:3$ 이고 다른 모델은 $3:7$ 이라고 합시다. 어떤 모델을 선택해야 할까요? 이럴 때는 **오컴의 면도날(Ockham's razor)**의 힘을 빌립니다. 위키피디아에 있는 '오컴의 면도날'의 원문은 다음과 같습니다.

> 1. "많은 것들을 필요없이 가정해서는 안된다" (Pluralitas non est ponenda sine neccesitate.)
> 2. "더 적은 수의 논리로 설명이 가능한 경우, 많은 수의 논리를 세우지 말라."(Frustra fit per plura quod potest fieri per pauciora.) - 위키피디아 : 오컴의 면도날

좀 더 쉽게 말하자면 *"같은 현상을 설명하는 주장들이 있다면 그 중 가장 가정이 적은(간단한) 것을 선택하라"* 라는 의미입니다. 결국 우리는 두 모델 중 복잡한, 즉 고려하는 특성이 많은 모델은 배제해주어야 합니다. 그렇기 때문에 편향의 비율이 더 높은 모델을 선택하는 것이 더 옳게 됩니다.




# Regularization

**정칙화(Regularization)**는 오버피팅을 방지하기 위한 하나의 방법입니다. 정칙화를 적용한 모델은 데이터에 민감하게 반응하지 않기 때문에 분산에 의한 오차를 줄일 수 있습니다. 수식 관점에서 보면 비용 함수에 정칙화와 관련된 항(Regularization term)을 추가하여 모델을 조정하게 됩니다.

[선형 회귀(Linear regression)](https://yngie-c.github.io/machine learning/2020/04/07/Linear_reg/)모델을 규제하는 방법에 대해서 알아봅시다. 선형 회귀 모델에 정칙화를 해주는 방법은 크게 두 가지로 나누어집니다. 첫 번째는 가장 많이 사용되는 **릿지(Ridge)** 입니다. 릿지는 L2 norm을 사용하여 규제를 가하므로 L2 정칙화라고도 부릅니다. 특정한 경우에는 L1 정칙화라고도 불리는 **라쏘(Lasso)**를 사용하기도 하며 두 방법을 혼합한 **엘라스틱넷(Elastic Net)**을 사용하기도 합니다. 

각 방법을 사용할 때 비용 함수 $C(\theta)$ 는 다음과 같이 변하게 됩니다.



$$
\begin{aligned}
\text{without Regulation : }C(\theta) &= \frac{1}{2} \sum^N_{n=0}(\text{Train}_n - g(x_n, \theta))^2 \\
\text{Ridge : }C(\theta) &= \frac{1}{2} \sum^N_{n=0}(\text{Train}_n - g(x_n, \theta))^2 + \frac{\alpha}{2} \Vert \theta\Vert^2 \\
\text{Lasso : }C(\theta) &= \frac{1}{2} \sum^N_{n=0}(\text{Train}_n - g(x_n, \theta))^2 + \alpha \Vert \theta\Vert
\end{aligned}
$$



릿지에 의해서 구해지는 파라미터 $\theta$ 가 어떻게 달라지는 지를 수식으로 알아보겠습니다. 비용 함수를 최소화하는 $\theta$ 를 구하는 것이 목적이므로 미분하여 0이 되는 점을 최소제곱법을 사용하여 구하면 됩니다.

Ridge에 의해 구해지는 $w$ 가 어떻게 달라지는지 수식으로 나타내 보자. 오차 함수를 최소화하는 $w$ 를 구하는 것이므로 $E(w)$ 를 $w$ 로 미분하여 0이 되는 지점을 구한다.



$$
\begin{aligned}
\frac{d}{d\theta}C(\theta) &= \frac{d}{d\theta}(\frac{1}{2}\Vert\text{Train} - X\theta \Vert ^2 + \frac{\alpha}{2} \Vert \theta\Vert^2)\\
&= \frac{d}{d\theta}(\frac{1}{2}(\text{Train} - X\theta)^T(\text{Train} - X\theta) + \frac{\alpha}{2}\theta^T\theta)\\
&= \frac{d}{d\theta}(\frac{1}{2}\text{Train}^T\text{Train} - X^T\theta \cdot \text{Train} + \frac{1}{2}X^TX \theta^T\theta + \frac{\alpha}{2} \theta^T\theta) \\
&= - X^T \cdot \text{Train} + X^TX \theta + \alpha \theta = 0\\
&\therefore \theta = (X^TX+\alpha I)^{-1} X^T \cdot \text{Train}
\end{aligned}
$$



$\alpha$ 는 정칙화에서 등장하는 하이퍼파라미터로 정칙화의 정도를 결정합니다. $\alpha$ 가 매우 커지게 되면 $X^TX+\alpha I \approx \alpha I$ 가 되어버립니다. 결국 파라미터 행렬은 $\theta = \frac{1}{\alpha} \cdot X^T \cdot \text{Train}$ 이 되어 모델이 단순해지는 것을 알 수 있습니다. 반대로 $\alpha$ 가 0에 가까우면 정칙화를 적용하지 않았을 때와 식이 같아지기 때문에 정칙화의 영향력이 사라지게 됩니다. 실제로는 여러 $\alpha$ 값을 대입했을 때 어떤 모델이 생성되는 지를 보고 가장 적합한 모델을 선택하게 됩니다. 다음은 릿지에 다양한 $\alpha$ 를 적용했을 때의 모델을 그래프로 나타낸 것입니다. $\alpha$ 가 커질수록 모델이 단순해지는 것을 볼 수 있습니다.

<p align="center"><img src="https://www.analyticsvidhya.com/wp-content/uploads/2016/01/4.-ridge-output.png" alt="ridge" style="zoom: 33%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/">analyticsvidhya.com</a></p>

릿지와 라쏘 모델을 적용했을 때의 차이는 $\theta$ 중 정칙화 대상이 되는 변수들, 즉 영향이 없는 특성값에 부여되는 변수가 어떤 방식으로 처리되느냐에 있습니다. 릿지는 이런 변수들의 값을 0에 가깝게 바꿉니다. 하지만 라쏘는 이런 변수들을 아예 0으로 만들어 특성을 사라지게 합니다. 그렇기 때문에 라쏘는 조금 더 과격하게 단순해지는데 이를 아래 이미지에서 확인할 수 있습니다. 이 이미지는 라쏘를 적용했을 때 다양한 $\alpha$ 에 따라 그래프가 어떻게 변화하는 지를 나타내고 있습니다.

<p align="center"><img src="https://www.analyticsvidhya.com/wp-content/uploads/2016/01/7.-lasso-output1.png" alt="lasso" style="zoom:33%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/">analyticsvidhya.com</a></p>

선형 회귀 뿐만 아니라 다른 모델에도 정칙화를 적용할 수 있습니다. 아래는 [로지스틱 회귀(Logistic regression)](https://yngie-c.github.io/machine learning/2020/04/19/Logistic-Regression/)에 정칙화를 적용하는 경우입니다. 아래 식은 비용 함수를 규제 항을 더해주는 것이 아니라 우도(Likelihood)에 정칙화 항을 빼주는 방식으로 정칙화를 적용하고 있습니다. 역시 $\alpha$ 값을 변화시켜 정칙화 정도를 조정하며, $\alpha$ 가 커질수록 강한 정칙화를 수행합니다.



$$
\text{argmax}_{\theta} \sum^m_{i=1} \log p(y_i | x_i, \theta) - \alpha R(\theta) \\
\begin{aligned}
\text{L1 : } R(\theta) &= \Vert\theta\Vert_1 = \sum^m_{i=1} \vert\theta_i\vert \\
\text{L2 : } R(\theta) &= \Vert\theta\Vert_2^2 = \sum^m_{i=1} \theta_i^2
\end{aligned}
$$



[서포트 벡터 머신(Support vector machine)](https://yngie-c.github.io/machine learning/2020/04/26/SVM/)에도 규제를 적용할 수 있습니다. 규제를 적용한 비용 함수 $f$ 를 수식으로 아래와 같이 나타낼 수 있습니다. $C$ 값을 변화시켜 규제의 정도를 조절합니다. 선형 회귀 모델, 로지스틱 회귀 모델의 정칙화와는 달리 하이퍼파라미터 $C$ 가 커질수록 규제의 정도는 약해집니다. $C = \frac{1}{2 \alpha n}$ 이므로 비용 함수에 더해지는 $\alpha$ 값이 커질수록 $C$ 값은 줄어드는 것을 볼 수 있습니다.



$$
\begin{aligned}
f &= \text{argmin}_{f \in H} \{\frac{1}{n} \sum^n_{i=1}V(y_i, f(x_i)) + \alpha \Vert f\Vert^2_{H}\} \\
V(&y_i, f(x_i)) = (1-yf(x))_{+} , (s)_{+} = \max(s,0) \text{ 이면}\\
f &= \text{argmin}_{f \in H} \{\frac{1}{n} \sum^n_{i=1} (1-yf(x))_{+} + \alpha \Vert f\Vert^2_{H}\} \\
f &= \text{argmin}_{f \in H} \{C \sum^n_{i=1} (1-yf(x))_{+} + \frac{1}{2}\Vert f\Vert^2_{H}\} \\
C &= \frac{1}{2\alpha n}
\end{aligned}
$$



아래는 $C$ 가 작은 값(강한 정칙화)일 때와 큰 값(약한 정칙화)일 때의 결정 경계가 어떻게 변화하는 지를 보여주는 이미지입니다. 강한 정칙화가 적용되었을 때에는 아웃라이어 값을 무시하는 것을 볼 수 있습니다. 맨 아래 그림은 추가적인 데이터가 들어왔을 때 정칙화의 효과를 보여줍니다. 약한 정칙화가 적용된 모델은 X표시된 몇 개의 인스턴스를 잘못 분류하는 것을 볼 수 있습니다

<p align="center"><img src="https://i.stack.imgur.com/GbW5S.png" alt="svm" style="zoom: 67%;" /></p>

<p align="center"><img src="https://i.stack.imgur.com/jfJ9G.png" alt="svm2" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel">stackexchange.com</a></p>