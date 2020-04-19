---
<>layout: post
title: 선형 회귀 (Linear Regression)
category: Machine Learning
tag: Machine-Learning
---

 본 포스트는 [카이스트 문일철 교수님의 강의](https://www.edwith.org/machinelearning1_17/joinLectures/9738) 를 바탕으로 작성하였습니다. 책은 [핸즈온 머신러닝](http://www.yes24.com/Product/Goods/59878826) 을 참고하였습니다.



# Linear Regression

어떤 데이터셋은 선형의 모양을 가지고 있다고 추측해볼 수 있다. 어떤 데이터가 선형이라는 가설을 세웠을 때 **선형회귀(Linear Regression)** 을 사용한다. 선형 회귀에서 기울기를 나타내는 파라미터를 $\theta_i \quad (단, i \geq 1)$ 라고 하고 편향(bias)를 나타내는 파라미터를 $\theta_0$ 라하면 속성값 $x_i$ 에 대하여 회귀식을 아래와 같이 나타낼 수 있다.



$$
h:\hat{f}(x;\theta) = \theta_0 + \sum_{i=1}^n \theta_i x_i = \sum_{i=0}^n \theta_i x_i
$$



여기서 속성값은 주어져 있기 때문에 가장 적절한 $\theta$ 를 찾는 것이 더 좋은 가설을 만들기 위한 방법이다. 

## Find $\theta$ , 최소제곱법

위에서 나타낸 가설에 대한 식을 행렬을 이용하여 좀 더 간단한 모양으로 나타낼 수 있다.



$$
h: \hat{f}(x;\theta) = \sum_{i=0}^n \theta_i x_i \rightarrow \hat{f} = X \theta \\
\text{여기서} \quad X = \left(\begin{array}{ccc} 1 \quad \cdots \quad x^1_n \\ \vdots \quad \ddots \quad \vdots \\ 1 \quad \cdots \quad x_n^D \end{array}\right), \theta = \left(\begin{array}{c} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{array}\right)
$$



[이전 게시물]([https://yngie-c.github.io/machine%20learning/2020/04/05/rule_based/](https://yngie-c.github.io/machine learning/2020/04/05/rule_based/)) 에서도 볼 수 있듯 실제의 세계에는 다양한 노이즈가 존재한다. 때문에 나타내는 함수 $f (\neq \hat{f})$ 는 에러에 대한 항 $e$ 을 가지고 있다.



$$
f(x;\theta) = \sum_{i=0}^n \theta_i x_i + e = y \rightarrow f = X \theta + e = Y
$$



이 노이즈로부터 발생하는 에러를 최소화해야만 한다. 그리고 이 에러는 실제 데이터와 가설 함수의 차이다. 두 함수의 행렬식을 조작하여 가장 적절한 $\theta$ 를 구할 수 있다.



$$
\hat{\theta} = \text{argmin}_\theta (e)^2 = \text{argmin}_\theta (f - \hat{f})^2 = \text{argmin}_\theta (Y - X\theta)^2 \\ = \text{argmin}_\theta (Y - X\theta)^T (Y - X\theta) = \text{argmin}_\theta (Y^T - \theta^TX^T) (Y - X\theta) \\ = \text{argmin}_\theta (\theta^TX^TX\theta - 2\theta^TX^TY + Y^T Y) = \text{argmin}_\theta (\theta^TX^TX\theta - 2\theta^TX^TY) \\
\because Y^T Y \text{ is constant}
$$



[이곳]([https://yngie-c.github.io/machine%20learning/2020/04/04/mle_map/](https://yngie-c.github.io/machine learning/2020/04/04/mle_map/))에서 MLE, MAE를 구했을 때처럼 미분을 통해 가장 적절한 $\theta$ 를 도출해낼 수 있다.



$$
\hat{\theta} = \text{argmin}_\theta (\theta^TX^TX\theta - 2\theta^TX^TY) \\ \text{미분이 0이 되는 }\theta \text{ 를 구한다.} \\
\nabla_\theta (\theta^TX^TX\theta - 2\theta^TX^TY) = 0 \\
X^TX\theta - 2X^TY = 0 \\
\theta = (X^TX)^{-1}X^TY
$$



위에서 최상의 $\theta$ 를 찾는 일련의 수학적인 과정을 **최소제곱법(Least Squared Method)** 이라고 하며 [이곳]([https://yngie-c.github.io/linear%20algebra/2020/03/09/LA4/](https://yngie-c.github.io/linear algebra/2020/03/09/LA4/)) 에서 좀 더 자세히 살펴볼 수 있다. 위키백과 [정규방정식(Normal Equation)]([https://ko.wikipedia.org/wiki/%EC%A0%95%EA%B7%9C%EB%B0%A9%EC%A0%95%EC%8B%9D](https://ko.wikipedia.org/wiki/정규방정식)) 을 참조하는 방법도 있다.

최소제곱법을 사용하여 선형 모델을 최적화하는 방법은 계산 복잡도가 크다는 단점을 가지고 있다. 정규방정식의 계산복잡도는 샘플의 수 $(m)$ 에 대해서는 $O(m)$으로 높지 않지만, 특성 수 $(n)$ 에 대해서는 일반적으로 $O(n^{2.4})$ 에서 $O(n^{3})$ 사이를 나타낸다. 수십 ~ 수백개의 특성을 가진 데이터셋을 가지고 최소제곱법으로 선형 모델 최적화를 진행하면 매우 오랜 시간과 큰 컴퓨터 자원을 사용하게 된다.



## Gradient Descent, 경사 하강법

**경사 하강법(Gradient Descent)** 은 최소제곱법을 사용하지 않고 최적점을 찾아가는 알고리즘이다. 비용 함수를 설정하고 이를 최소화하는 방향으로 반복해서 파라미터를 조정해 나간다. 파라미터 벡터 $\theta$ 에 임의의 초깃값을 주는 것에서부터 알고리즘을 시작한다. 그리고 최솟값에 수렴할 때까지 점진적으로 향상시켜 나가게 된다.

경사하강법에서 중요한 하이퍼파라미터는 스텝의 크기라고도 할 수 있는 **학습률(Learning rate, $\eta$ )** 이다. 학습률을 너무 작게 설정하면 수렴하기 위해 반복을 많이 진행해야 하므로 시간이 오래 걸리게 되거나 오랜 반복 끝에도 최적점에 이르지 못하게 된다. 반대로 학습률이 너무 크면 최적점을 지나쳐버리는 바람에 적절한 수렴점을 찾지 못하게 된다.

**배치 경사 하강법(Batch Gradient Descent)** 은 가장 일반적인 경사 하강법 방식이다. 모델의 속성에 따른 파라미터 $\theta_i$ 에 대한 비용 함수의 그래디언트를 계산하는 과정이다. 그래디언트 계산을 위해 **편도함수** (Partial Derivative)를 구한다. 편도함수는 아래의 수식을 통해 구할 수 있다.



$$
\frac{\partial}{\partial \theta_j}MSE(\theta) = \frac{2}{m}\sum_{i=1}^{m}(\theta^T \cdot \mathbf{x}^{(i)} - y^{(i)})x^{(i)}_j
$$



이를 그래디언트 벡터를 통해 한번에 나타내면 다음과 같다.



$$
\nabla_\theta MSE(\theta) = \left[\begin{array}{ccc} \frac{\partial}{\partial \theta_0}MSE(\theta) \\ \frac{\partial}{\partial \theta_1}MSE(\theta) \\ \vdots \\ \frac{\partial}{\partial \theta_n}MSE(\theta) \end{array}\right] = \frac{2}{m}\mathbf{X}^T \cdot (\mathbf{X} \cdot \theta - \mathbf{y})
$$



이렇게 그래디언트 벡터를 구한 뒤에는 학습률을 곱하여 이전 $\theta$ 에서 빼주어 새로운 $\theta$ 로 개선해 나간다. 그래디언트 벡터와 이 과정을 아래와 같은 수식으로 나타낼 수 있다.



$$
\theta^{d+1} = \theta^d - \eta \cdot \nabla_\theta MSE(\theta)
$$



배치 경사 하강법에서는 매번 모든 데이터셋에 대해 그래디언트를 계산한다. 때문에 데이터셋의 크기가 커지면 시간이 오래 걸리고 컴퓨팅 자원이 많이 소모된다. **확률적 경사 하강법(Stochastic Gradient Descent, SGD)** 은 이런 문제를 개선하기 위한 알고리즘이다. 매 스텝마다 한 개의 샘플을 무작위로 선택한 뒤 하나의 샘플에 대해 그래디언트를 계산해 나간다. 반복마다 사용되는 데이터의 양이 적기 때문에 훨씬 빠른 속도로 알고리즘을 진행할 수 있다. 하지만 샘플을 랜덤하게 선택한다는 특성 때문에 배치 경사 하강법보다 불안정하다는 문제가 있다. **미니배치 경사 하강법(Mini-batch Gradient Descent)** 은 전체 데이터셋 혹은 하나의 데이터 대신 작은 샘플 세트를 구성하여 경사하강법 알고리즘을 진행하는 방법이다. 지금까지 살펴본 모든 알고리즘을 아래의 표로 나타낼 수 있다.

| 알고리즘             | 샘플 수가 클 때 | 특성 수가 클 때 | 외부 메모리 학습 지원 | 스케일 조정 필요 | 하이퍼 파라미터 수 |
| -------------------- | --------------- | --------------- | --------------------- | ---------------- | ------------------ |
| 정규방정식           | 빠름            | 느림            | X                     | X                | 0                  |
| 배치 경사 하강법     | 느림            | 빠름            | X                     | O                | 2                  |
| 확률적 경사 하강법   | 빠름            | 빠름            | O                     | O                | >=2                |
| 미니배치 경사 하강법 | 빠름            | 빠름            | O                     | O                | >=2                |



## Polynomial, 다항 회귀

데이터셋이 직선보다 복잡한 형태를 보일 때가 있다. 이런 비선형 데이터를 학습하는 데 선형 모델을 사용할 수 있다. 각 특성의 거듭제곱을 새로운 특성으로 추가하여 선형 모델을 훈련시킨다. 이를 **다항 회귀(Polynomial Regression)** 이라고 한다. 이처럼 많이 차원을 높여 고차 다항 회귀를 적용하면 선형 회귀보다 훨씬 더 훈련 데이터에 잘 맞게 된다. 하지만 이런 모델은 새로운 데이터에는 적합하지 않은 결과를 보여주는 과적합(Overfitting)을 유발하기 쉽다. 이를 방지하기 위해서 머신러닝에서는 **교차 검증(Cross validation)**[^1] 을 통한 **학습 곡선(Learning curve)**[^2] 을 사용한다. 



[^1]: 폴드(Fold)라고 불리는 서브셋으로 분할한 후 그 중 하나를 검증 세트로 만든다. K개로 나누면 아래 그림과 같이 K개의 데이터셋이 만들어진다. 때문에 K-Fold 교차 검증이라고도 불린다. 이미지 출처 : [네이버 블로그](https://m.blog.naver.com/PostView.nhn?blogId=dnjswns2280&logNo=221532535858&categoryNo=17&proxyReferer=https:%2F%2Fwww.google.com%2F)

<p align='center'><img src="https://cdn-images-1.medium.com/max/1600/1*rgba1BIOUys7wQcXcL4U5A.png" alt="교차 검증" style="zoom:30%;" /></p>



[^2]: 학습 세트와 검증 세트의 학습(검증) 결과를 그래프로 나타낸 것이다. 어떤 지점에서 검증 데이터셋이 최대가 되는지 혹은 과적합이 발생하는지 등을 시각화하여 볼 수 있다.

