---
layout: post
title: 과적합과 과소적합 (Overfitting & Underfitting)
category: Machine Learning
tag: Machine-Learning
---

 본 포스트는 [카이스트 문일철 교수님의 강의](https://www.edwith.org/machinelearning1_17/joinLectures/9738) 를 바탕으로 작성하였습니다. 책은 [핸즈온 머신러닝](http://www.yes24.com/Product/Goods/59878826) 을 참조하였습니다.



# Bias & Variance

## Train Data and Test Data

우리의 목적은 데이터를 학습시켜 기계가 더 정확한 모델을 만들어내도록 하는 것이 목적이다. ([PAC Learning](http://sanghyukchun.github.io/66/) 에 대한 내용을 다시 상기해볼 수 있다.) 이 때 더 정확한 모델을 만들어내기 위해 사용하는 데이터셋을 **학습 데이터셋 혹은 훈련 데이터셋(Train dataset)** 이라고 한다. 하지만 학습 데이터를 사용해서 만들어진 모델이 새로운 데이터에도 잘 맞으리라는 보장은 없다. 그래서 우리는 모델을 출시하기 전에 테스트를 해야한다. 학습에 사용된 데이터를 테스트에는 사용할 수 없다. 때문에 가지고 있는 데이터셋을 학습용과 테스트 용으로 나누어야한다. 이렇게 테스트용으로 분리되는 데이터를 **테스트 데이터셋(Test dataset)** 이라고 한다. 일반적으로 전체 데이터셋의 15~20% 정도가 Test dataset으로 분류된다.



## Overfitting & Underfitting

우리가 모델을 생성하는 방법에 따라서 다양한 모델이 생성될 수 있다. 아래 그림은 어떤 데이터가 주어졌을 때 다항선형회귀(Polynomial Linear regression)를 사용하여 생성한 모델의 그래프이다. 왼쪽부터 각각 1차, 2차, N차 다항 회귀를 사용하였다.

![over vs under](http://sanghyukchun.github.io/images/post/59-1.png)

<p align="center" style="font-size:80%">이미지 출처 : 문일철 교수님 강의 학습자료</p>

먼저 가운데 모델부터 살펴보면 데이터셋의 처음부터 끝까지 데이터의 모양을 따라 잘 근사하고 있음을 알 수 있다. 학습 데이터로부터 발생하는 약간의 오차가 있지만 전체 데이터의 모양에서 벗어나는 경우는 없어보인다. 다음으로 왼쪽에 그려진 선형 그래프는 차수를 낮추어 단순화한 모델이다. 이 때 우상향 하는 데이터의 모습은 잘 근사하고 있으나 왼쪽 부분의 데이터는 제대로 근사하고 있지 못하는 듯 하다. 마지막으로 오른쪽에 그려진 그래프는 차원을 많이 높여 복잡하게 근사한 모델이다. 이미지에서 나타나는 학습 데이터와 모델의 오차는 거의 없어 보인다. 하지만 이렇게 생성된 모델의 경우 새로운 데이터가 외부에서 들어왔을 때 모델과의 오차가 매우 커질 수 있다. 예를 들어 실제 데이터 중에 그래프에서 가장 y값이 높게 나타난 데이터와 비슷한 경우가 많을 경우 오차는 가운데 모델보다 훨씬 더 커지게 된다.

왼쪽처럼 너무 단순한 모델을 생성하여 훈련 데이터와 잘 맞지 않는 경우를 **과소적합(Underfitting)** 이라고 하며 오른쪽처럼 너무 복잡한 모델을 생성하여 테스트 데이터 혹은 새로운 데이터와 잘 맞지 않는 것을 **과(대)적합(Overfitting)** 이라고 한다. 전자(훈련 데이터와 모델의 값)에서 발생하는 오류를 **편향(Bias)** 이라고 하고 후자(새로운 데이터를 관측하지 못함)에서 발생하는 오류를 **분산(Variance)** 이라고 한다. 이 편향과 분산의 트레이드 오프(Bias-Variance Trade-off)를 잘 조정해야 더 좋은 모델을 생성할 수 있다.



## Bias-Variance Trade-off

우리가 생성한 모델에서 발생하는 모든 오류를 $E_{\text{out}}$ 이라고 하면 이를 알고리즘을 학습 과정(Approximation)에서 발생하는 오류와 모든 데이터를 관측하지 못하는 데(Generalization)서 발생하는 오류로 나눠 볼 수 있다. 이를 수식으로 아래와 같이 나타낼 수 있다.


$$
E_{\text{out}} \leq E_{\text{in}} + \Omega \\
E_{\text{in}} : \text{모델이 근사되는 과정에서 발생하는 오류} \\
\Omega : \text{모든 데이터를 관측하지 못하는 데서 발생하는 오류}
$$


$f$ 를 최적의 목적 함수, $g$ 를 ML이 학습하여 생성한 함수라고 하고 존재하는 모든 (관측 범위를 넘어서는) 데이터 $D$ 에 대해서 생성되는 ML 모델 함수를 $g^{D}$ 라고 하자. 이 중 무한개의 데이터셋을 추출하여 만든 ML 모델 함수의 평균을 $\bar{g}$ 라고 나타내도록 한다. 이 때 $\bar{g}(x) = E_D[g^{D}(x)]$ 로 나타낼 수 있다.

하나의 인스턴스로부터 발생하는 오류는 다음과 같이 나타낼 수 있다.


$$
E_{\text{out}}(g^D (x)) = E_{X}[(g^D (x) - f(x))^2]
$$


이를 무한 개의 데이터셋으로부터 발생하는 오류의 기댓값으로 확장하면 아래와 같다. 


$$
E_D[E_{\text{out}}(g^D (x))] = E_D[E_X[(g^D (x) - f(x))^2]] = E_X[E_D[(g^D (x) - f(x))^2]]
$$


가장 오른쪽 식 안에 있는 $E_D[(g^D (x) - f(x))^2]$ 부분을 적절히 조작하면 다음과 같이 단순화 할 수 있다.


$$
\begin{aligned}
E_D[(g^D (x) - f(x))^2] &= E_D[(g^D (x) - \bar{g}(x) + \bar{g}(x) - f(x))^2]\\
&= E_D[(g^D (x) - \bar{g}(x))^2 + (\bar{g}(x) - f(x))^2 + 2(g^D (x) - \bar{g}(x))(\bar{g}(x) - f(x))] \\
&= E_D[(g^D (x) - \bar{g}(x))^2] + (\bar{g}(x) - f(x))^2 + E_D[2(g^D (x) - \bar{g}(x))(\bar{g}(x) - f(x))] \\
&= E_D[(g^D (x) - \bar{g}(x))^2] + (\bar{g}(x) - f(x))^2\\
&\because E_D[2(g^D (x) - \bar{g}(x))(\bar{g}(x) - f(x))] = 0
\end{aligned}
$$


 따라서 무한 개의 데이터셋으로부터 발생하는 오류의 식은 아래처럼 변형할 수 있다.


$$
E_D[E_{\text{out}}(g^D (x))] = E_X[E_D[(g^D (x) - \bar{g}(x))^2] + (\bar{g}(x) - f(x))^2]
$$


여기에서 첫 번째 항, 즉 모든 데이터를 관측하지 못하는 데서 오는 오차는 분산(Variance)에 해당하는 항이다. 그리고 두 번째 항은 타깃 함수와 ML이 생성한 함수가 같지 않는 데서 발생하는 오류인 편향(Bias)의 제곱을 나타낸다. 즉 분산을 줄이려면 더 많은 데이터를 모아야 하고 편향을 줄이려면 주어진 데이터에 더 잘 맞는(복잡한) 모델을 만들어야 한다. 그러나 편향을 줄이려 하면 분산이 늘어나고 분산을 줄이려 하면 편향이 늘어나는데 이를 **편향과 분산의 딜레마(Bias & Variance Dilemma)** 라고 한다.

그렇다면 특정 데이터셋에서 오류는 같지만 편향과 분산은 다른 두 모델이 생성되었다고 하자. 두 모델의 오류는 모두 10이지만 한 모델은 편향 2, 분산 8이고 다른 모델은 편향 8, 분산 2라고 하자. 둘 중 어떤 모델을 선택하는 것이 좋을까? 이럴 때는 **오컴의 면도날(Occam's Razor)** 의 힘을 빌린다. 오컴의 면도날이란 "여러 가설이 경쟁중일 때 그 중 가장 가정이 적은 가설이 채택되어야 한다"는 의미다. 여기서 가장 가정이 적다는 이야기는 덜 복잡한(간단한) 모델이라는 이야기다. 즉 오류가 비슷한 두 모델이 있을 경우에는 분산이 적은 모델을 선택하는 것이 옳다.



## Cross Validation

실제로는 $\bar{g}$ 처름 무한한 데이터셋을 뽑아낼 수도 없다. 하지만 특정 기법을 사용하여 데이터 셋의 개수를 늘리는 방법으로 따라해볼 수는 있다. 그 기법을 **K-겹 교차검증(K-fold Cross Validation)** 이라고 한다. 일단 가지고 있는 데이터셋을 K개의 서브셋으로 나눈다. 그 중 K-1개의 서브셋을 학습에 사용하고 1개의 서브셋을 **검증(Validation)** 에 사용한다. 



## Metrics

우리는 목적 함수 $f(x)$ 와 가설의 평균 함수인 $\bar{g}(x)$ 를 알지 못하기 때문에 편향과 분산을 각각 구할 수는 없다. 때문에 실제로 모델의 성능을 측정할 때에는 다른 수치를 사용한다. 이런 수치로는 Accuracy, Precision, Recall, F-Measure, ROC curve(AUC) 등이 있다. 각각의 수치를 알아보기 위해서 먼저 **Confusion Matrix** 에 대해 알아보자.

Confusion Matrix 는 각 데이터의 실제 클래스와 예측 클래스의 개수를 표로 나타낸 것이다. 물론 그리는 방법에 따라 다르지만 아래의 경우 각 행(Row)은 실제 클래스를 나타내고 각 열(Column)은 예측 클래스를 나타내고 있다.  

<p align="center" style="font-size:80%"><img src="https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg" alt="CM" style="zoom: 80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://manisha-sirsat.blogspot.com/">manisha-sirsat.blogspot.com</a></p>

Confusion Matrix로부터 **정확도(Accuracy), 정밀도(Precision)** , **재현율(Recall)** 혹은 민감도(Sensitivity), F-measure 등을 구할 수 있다. 먼저 정확도(Precision)와 재현율(민감도)은 다음과 같이 나타낼 수 있다.


$$
\begin{aligned}
\text{Accuracy} &= \frac{\text{TP+TN}}{\text{TP + TN + FP + FN}} : \text{전체 데이터 중 제대로 판별한 데이터의 비율} \\
\text{Precision} &= \frac{\text{TP}}{\text{TP + FP}} : \text{Positive로 판별한 것 중 실제로 Positive인 데이터의 비율}\\
\text{Recall(Sensitivity)} &= \frac{\text{TP}}{\text{TP + FN}} : \text{실제 Positive 데이터 중 Positive로 판별된 데이터의 비율}
\end{aligned}
$$


정밀도와 재현율은 둘 다 중요하다. **F-Measure(혹은 F-score)** 를 사용하면 두 수치를 한 번에 볼 수 있다. F-Measure 중 가장 많이 사용되는 $\text{F}_1 \text{-Score}$ 는 아래와 같이 나타낼 수 있다. 


$$
\text{F}_1 \text{-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$


특정 상황에서는 정밀도가 재현율보다 부각되어야 하거나 재현율이 정밀도보다 강조되어야 하는 경우가 발생한다. 예를 들어 Spam메일을 분류할 때는 Negative case(스팸 메일이 아닌 메일 = 중요한 메일이 있을 수도)가 Positive로 분류되어서는 안되기 때문에 정밀도가 훨씬 강조된다. VIP 고객을 분류할 때는 Positive case(VIP 고객)가 Negative로 분류되어서는 안되므로 재현율이 훨씬 강조된다. 이럴 때는 경우에 맞추어 $\text{F}_\text{b} \text{-Score}$ 를 사용한다.


$$
\text{F}_\text{b} \text{-Score} = \frac{(1+b^2) \times \text{Precision} \times \text{Recall}}{b^2 \times \text{Precision} + \text{Recall}} \\
\text{Precision 이 강조되는 경우 (b < 1) Ex)} \text{F}_\text{0.5} \text{-Score} \\
\text{Recall 이 강조되는 경우 (b > 1) Ex)} \text{F}_\text{2} \text{-Score} \\
\text{F}_\text{1} \text{-Score 는} \text{F}_\text{b} \text{-Score 의 특수한 케이스}
$$


# Regulation

데이터 셋을 추출할 때 정말 이상하게 샘플링되는 경우가 있을 수 있다. 이 때 분산(Variance)에 의해 발생하는 오류는 매우 크다. 이런 사태를 대비하기 위해 완벽한 Fitting을 포기하는 방향으로 모델을 설계하기도 한다. 분산에 의한 오류, 즉 과적합(Overfitting)을 막기 위한 두 가지 방법이 있다. 첫 번째는 위에서 살펴보았던 간단한(저차원의) 모델을 사용하는 것이다. 하지만 무조건 간단한 모델을 선택할 경우 편향(Bias)에 의해서 발생하는 에러가 증가한다. 이 때문에 두 번째 방법인 **규제(Regulation)** 가 고안되었다. 규제란 모델이 생성되면서 데이터에 민감하게 반응하지 않도록 막는다. 오류(Error) 함수에 규제와 관련된 새로운 항(Regulation term)을 추가하여 모델을 조정할 수 있다.

이 때 페널티를 주는 방식에 따라 규제항의 모습이 달라진다. 여러 방식 중에서 일반적으로는 **릿지(Ridge, L2 Regulation)** 를 가장 많이 사용한다. 몇 가지 경우에 있어서는 **라쏘(Lasso, L1 Regulation)** 를 사용할 때도 있으며 두 가지 방법을 혼합한 **엘라스틱넷(Elastic Net)** 을 사용하기도 한다. 각 방법을 사용할 때의 Error 함수는 다음과 같다.


$$
\begin{aligned}
\text{Ridge : }E(w) &= \frac{1}{2} \sum^N_{n=0}(\text{Train}_n - g(x_n, w))^2 + \frac{\lambda}{2} \vert\vert w\vert\vert^2 \\
\text{Lasso : }E(w) &= \frac{1}{2} \sum^N_{n=0}(\text{Train}_n - g(x_n, w))^2 + \lambda \vert\vert w\vert\vert
\end{aligned}
$$


Ridge에 의해 구해지는 $w$ 가 어떻게 달라지는지 수식으로 나타내 보자. 오류 함수를 최소화하는 $w$ 를 구하는 것이므로 $E(w)$ 를 $w$ 로 미분하여 0이 되는 지점을 구한다.


$$
\begin{aligned}
\frac{d}{dw}E(w) &= \frac{d}{dw}(\frac{1}{2}||\text{Train} - Xw||^2 + \frac{\lambda}{2}||w||^2)\\
&= \frac{d}{dw}(\frac{1}{2}(\text{Train} - Xw)^T(\text{Train} - Xw) + \frac{\lambda}{2}w^Tw)\\
&= \frac{d}{dw}(\frac{1}{2}\text{Train}^T\text{Train} - X^Tw \cdot \text{Train} + \frac{1}{2}X^TXw^Tw + \frac{\lambda}{2}w^Tw) \\
&= - X^T \cdot \text{Train} + - X^TXw + \lambda w = 0\\
&\therefore w = (X^TX+\lambda I)^{-1} X^T \cdot \text{Train}
\end{aligned}
$$


여기서 $\lambda$ 는 규제의 정도를 결정하는 하이퍼 파라미터이다. $\lambda$ 가 0에 너무 가까우면 데이터에 민감하게 반응하기 때문에 오버피팅이 발생한다. 반대로 $\lambda$ 가 지나치게 높으면 어떤 데이터가 들어오더라도 간단한 모델을 생성하기 때문에 차원을 늘린 후 규제를 가하는 의미가 사라진다. 실제로는 여러 $\lambda$ 를 대상으로 실험해보면서 적절한 $\lambda$ 값을 찾을 수 있다.

## Regulation for other Models

규제는 선형 회귀뿐만 아니라 다른 모델에도 적용할 수 있다. 로지스틱 회귀에 규제를 적용할 경우 확률 함수는 다음과 같이 나타난다. $\alpha$ 값을 변화시켜 규제의 정도를 조정한다. $\alpha$ 가 커질수록 규제의 정도가 강해진다.


$$
\text{argmax}_{\theta} \sum^m_{i=1} \log p(y_i | x_i, \theta) - \alpha R(\theta) \\
\begin{aligned}
\text{L1 : } R(\theta) &= ||\theta||_1 = \sum^m_{i=1} |\theta_i| \\
\text{L2 : } R(\theta) &= ||\theta||_2^2 = \sum^m_{i=1} \theta_i^2
\end{aligned}
$$


SVM(Support Vector Machine) 에도 규제가 적용된다. 규제를 적용한 에러함수 $f$ 는 다음과 같다. $C$ 값을 변화시켜 규제의 정도를 조정한다. $C$ 가 커질수록 규제의 정도가 강해진다.


$$
\begin{aligned}
f &= \text{argmin}_{f \in H} \{\frac{1}{n} \sum^n_{i=1}V(y_i, f(x_i)) + \lambda||f||^2_{H}\} \\
V(y_i, &f(x_i)) = (1-yf(x))_{+} , (s)_{+} = \max(s,0) \text{ 이면}\\
f &= \text{argmin}_{f \in H} \{\frac{1}{n} \sum^n_{i=1} (1-yf(x))_{+} + \lambda||f||^2_{H}\} \\
f &= \text{argmin}_{f \in H} \{C \sum^n_{i=1} (1-yf(x))_{+} + \frac{1}{2}||f||^2_{H}\} \\
C &= \frac{1}{2\lambda n}
\end{aligned}
$$

