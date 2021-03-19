---
layout: post
title: 랜덤 포레스트(Random Forest)
category: Machine Learning
tag: Machine-Learning
---



해당 게시물은 [고려대학교 강필성 교수님의 강의](https://github.com/pilsung-kang/Business-Analytics-IME654-)를 바탕으로 작성한 것입니다.

# Random Forest

**랜덤 포레스트(Random Forest)**는 자주 사용되는 앙상블 기법 중 하나입니다. 기본 모델로 의사 결정 나무(Decision tree)를 사용하기 때문에 배깅(Bagging)의 특수한 형태입니다. 배깅은 기본 모델로 무엇을 설정하든 상관없지만 랜덤 포레스트는 의사 결정 나무를 사용해야 합니다. 다수의 의사 결정 나무(Tree)를 결합한 모델이기에 Forest라는 이름이 붙었습니다.

<img src="https://images.unsplash.com/photo-1425913397330-cf8af2ff40a1?ixid=MXwxMjA3fDB8MHxzZWFyY2h8M3x8Zm9yZXN0fGVufDB8fDB8&amp;ixlib=rb-1.2.1&amp;auto=format&amp;fit=crop&amp;w=500&amp;q=60" alt="forest" style="zoom:80%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://unsplash.com/">unsplash.com</a></p>



## Why Random?

그렇다면 Forest 앞에 붙은 Random은 무슨 의미일까요? 변수를 임의(Random)로 선택한다는 뜻인데요. 이것이 랜덤 포레스트와 의사 결정 나무를 사용한 배깅의 차이점입니다. 원래의 [의사 결정 나무](https://yngie-c.github.io/machine%20learning/2020/04/06/decision_tree/)는 결정 경계를 탐색할 때 모든 변수를 고려하여 정보 획득량(Information Gain, IG)이 가장 높은 지점을 찾아냅니다. 랜덤 포레스트는 이 중 일부의 변수만을 선택하여 정보 획득량이 최고인 지점을 탐색합니다.

아래는 25개의 변수 중에서 임의로 5개의 변수를 택할 때 변수가 어떻게 선택되는 지를 나타낸 그림입니다.

![variable_selection](https://user-images.githubusercontent.com/45377884/111717707-f6fcc080-889b-11eb-98ef-8736f8b5f4c9.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/Business-Analytics-IME654-">github.com/pilsung-kang/Business-Analytics-IME654</a></p>

첫 번째 결정 경계를 만들 때에는 임의로 5개의 특성 $(x_2, x_{10}, x_{23}, x_{17}, x_9)$ 이 선택되었습니다. 하지만 Depth가 2일 때에는 각각 $(x_1, x_{13}, x_{21}, x_{11}, x_7), (x_{24}, x_{15}, x_2, x_7, x_{19})$ 이 선택된 것을 확인할 수 있습니다.

### Effect

이렇게 변수를 랜덤하게 선택하는 이유는 무엇일까요? 오히려 의사 결정 나무의 성능이 떨어질 텐데 말이지요. 그 이유는 **다양성**에 있습니다. 배깅만을 적용한 트리는 데이터 셋은 조금 다를 수 있어도 모든 변수를 탐색하기 때문에 비슷한 모델이 생성될 확률이 매우 높습니다. 하지만 랜덤 포레스트는 매번 다른 특성을 고려하기 때문에 훨씬 더 다양한 모델을 만들어낼 수 있습니다.

이해를 도울 수 있는 예시를 들어보겠습니다. 아래는 A반과 B반 학생 3명의 성적을 나타낸 표입니다.

![example](https://user-images.githubusercontent.com/45377884/111719948-48a74a00-88a0-11eb-8517-fa50fb42717c.png)

학생 각각의 평균은 A반이 월등히 높습니다. B반 학생들의 평균은 모두 72점인데 A반 학생들의 평균은 약 90점이나 됩니다. 하지만 3명이 머리를 맞대고 국어, 영어, 수학을 풀어야 하는 상황이라면 결과는 조금 다릅니다. 각 과목의 최고점을 모아보면 A반은 $(91, 92, 94)$ 점이고 B반은 $(96, 96, 96)$ 점입니다. 개개인의 퍼포먼스는 A반이 더 높지만 집단 지성의 관점에서 봤을 때에는 B반의 퍼포먼스가 더 높겠네요.

완전히 동일한 예시는 아니겠습니다만 랜덤 포레스트가 변수를 선택하여 사용하는 이유도 이와 유사합니다. 특정 변수를 제약함으로써 전체적인 성능은 떨어지더라도 몇몇 특성에 전문화된 의사 결정 나무를 최대한 많이 만들지요. 그리고 이들을 한데 모았을 때의 시너지 효과를 기대하는 것입니다.

## Generalization Error

랜덤 포레스트는 트리의 개수(Population size)가 일정 수준 이상으로 커졌을 때 **일반화 오차(Generalization error)**를 구할 수 있습니다. 일반화 오차의 상한선(Upper bound)는 다음과 같이 결정됩니다.


$$
\text{Generalization error} \leq \frac{\bar{\rho}(1-s^2)}{s^2}
$$


위 식에서 $\bar{\rho}$ 는 각 트리 사이의 상관계수를 구한 뒤 평균을 구한 값입니다. $s^2$ 는 마진값으로, 이진 분류에서는  $\vert P(y=1) - P(y=0) \vert$ 이 됩니다. 개별 트리가 정확할수록 $s^2$ 이 커지기 때문에 일반화 오차는 감소합니다. 그리고 트리 사이의 상관관계가 적을수록 $\bar{\rho}$ 가 줄어들어 일반화 오차가 감소하게 됩니다.

## Permutation Importance

랜덤 포레스트에서는 **특정 변수의 중요도를 측정**할 수 있습니다. 변수의 중요도를 측정하는 방식은 다음과 같습니다. 먼저 아무것도 건드리지 않을 때의 검증 오차(OOB Error)를 측정합니다. 이 값을 $e_i$ 로 나타내겠습니다. 중요도를 측정하고자 하는 변수를 섞어준(Permutation) 뒤에 검증 오차를 측정합니다. 이 값을 $p_i$ 로 나타내겠습니다. 

즉, $p_i$ 는 해당 변수에 해당하는 값을 엉망으로 만든 뒤에 측정한 값입니다. 결정 경계를 형성할 때 해당 변수를 많이 사용했다면 $p_i$ 와 $e_i$ 의 차이가 커지겠지요. 그래서 모든 트리에 대해 $p_i - e_i$ 의 값을 구해주어 중요도를 측정합니다.

$m$ 번째 트리에서 두 값의 차이를 $d^m_i = p^m_i - e^m_i$ 라 하면 $i$ 번째 변수가 중요할수록 $d_i$ 의 평균은 증가하고, 분산은 작아집니다. 따라서 아래와 같이 해당 변수의 중요도 $v_i$ 를 구할 수 있습니다.


$$
\bar{d}_i = \frac{1}{m}\sum^m_{i=1}d_i^m, \quad s_i^2 = \frac{1}{m-1}\sum^m_{i=1}(d_i^m - \bar{d}_i)^2 \\
\therefore v_i = \frac{\bar{d}_i}{s_i^2}
$$


이렇게 구한 $v_i$ 는 변수의 상대적인 중요도를 나타낼 뿐 절대적인 의미는 없습니다. $v_i$ 값이 높은 변수가 다른 변수에 대해 상대적으로 중요할 뿐이지요. 아래는 특정 데이터의 중요도를 구한 뒤 시각화한 예시입니다.

<img src="https://www.researchgate.net/profile/Myat-Aung-3/publication/328307936/figure/fig5/AS:802070386966529@1568239890316/Feature-importance-20-most-important-features-for-logistic-regression-left-and-random.jpg" alt="feature_importance" style="zoom:50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.researchgate.net/figure/Feature-importance-20-most-important-features-for-logistic-regression-left-and-random_fig5_328307936">www.researchgate.net</a></p>

## Conclusion

의사 결정 나무를 활용한 배깅에 랜덤 변수 선택을 더하여 다양성을 극대화하는 **랜덤 포레스트(Random forest)**에 대해 알아보았습니다. 다음부터는 데이터셋이나 변수 설정에서 다양성을 확보하는 방법이 아닌 먼저 생성된 모델의 추정치로부터 또 다른 모델을 생성하여 다양성을 확보하는 부스팅(Boosting) 계열의 알고리즘에 대해서 알아보도록 하겠습니다.
