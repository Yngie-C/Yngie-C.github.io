---
layout: post
title: 차원 축소 - 특성 선택(Feature selection)
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# 차원 축소(Dimensionality Reduction)

## 차원 축소(Dimensionality Reduction)

Bag of Words 에서 나타나는 TDM(Term-Document Matrix)의 문제는 데이터의 차원이 너무 크다는 것이다. T의 개수는 일반적으로 D보다 훨씬 크게 나타난다. 일반적으로 통계학에서 관측치의 수가 최소한 변수의 수보다는 많아야 다중공선성 등 다양한 통계치를 만족한다는 가정이 가능하다. 하지만 위와 같이 변수(Term)의 수가 객체(Document)의 수보다 훨씬 더 많아진다면 일반적으로 사용되는 통계 가정을 적용하기 어렵다는 난점이 있다. 데이터의 Sparseness도 문제이다. 특정 단어가 등장하지 않는 문서가 많으므로 행렬 대부분의 요소가 0으로 나타난다. 이런 경우 너무 많은 컴퓨팅 자원이 사용되는 문제가 있다.

**차원 축소(Dimensionality Reduction)** 를 해주는 이유는 계산을 더 효율적으로 만들기 위해서이다. 뿐만 아니라 가능하다면 분류에서의 정확도와 군집에서의 모듈성, 동일한 결과를 달성하기 위한 학습 데이터를 줄이는 등의 퀄리티를 향상시키는 효과를 가져오기도 한다. 

차원 축소를 크게 **특성 선택(Feature Selection)** 과 **특성 추출(Feature Extraction)** 으로 나눌 수 있다. **특성 선택** 은 전체 특성 중에서 유의미한 부분 집합 특성을 선택하는 것으로 Filter와 Wrapper의 방식이 있다. 전자는 One-time으로 특정 기준을 만족하는 설명 변수를 선택하는 과정이다. 후자는 반복적인 피드백을 통해서 최적의 중요 변수 조합을 찾아나간다. **특성 추출** 은 원래의 변수를 활용하여 새로운 특성 집합을 만들어낸다. 특성 추출에는 주성분분석(Principal Component Analysis, PCA)으로 대표되는 최대 분산 방법, 다차원스케일링(Multi-Dimensional Scaling, MDS)으로 대표되는 최대 거리정보 보존법, 잠재의미분석(Latent Semantic Analysis, LSA)으로 대표되는 잠재구조 파악 방법 등이 있다. 특성 추출은 모두 비지도학습 방법을 사용하기 때문에 Class의 Label 정보를 사용하지 않는다.



# 특성 선택(Feature Selection)

아래 이미지에 있는 10 X 10 Binary TDM 예제 데이터를 보자. 이 데이터는 Positive/Negative의 이진분류 문제이며, 6개의 Positive 문서와 4개의 Negative 문서로 이루어져 있다. 이 예제 데이터로부터 각 특성(Term)의 중요도를 판단할 수 있는 10개의 수치(Metric)을 알아보도록 하자.

<이미지>

첫 번째 수치는 DF(Document Frequency)이다. DF는 측정하기 매우 쉽지만 Term 1, 2, 3의 예시에서 알 수 있듯 변별력이 그다지 높지 않다. DF를 구하는 식은 아래와 같다.


$$
\text{DF}(w) = N_D(w)
$$


두 번째 수치는 정확도(Accuracy, ACC)이다. 정확도도 DF만큼이나 측정하기가 쉽지만 Negative 함수에만 등장하는 Term 2의 경우에 음수 때문에 수치가 낮아지는 문제가 발생한다. ACC를 측정하는 수식은 아래와 같다.


$$
\text{Acc}(w) = N(\text{Pos},w) - N(\text{Neg}, w)
$$


정확도에서 발생하는 문제를 해결하기 위한 것이 정확도 비율(Accuracy Ratio)이다. 정확도 비율을 구하는 식은 아래와 같으며 Term 1, 2, 3에 대하여는 위 수치들보다 더 잘 예측하고 있음을 알 수 있다.


$$
\text{AccR}(w) = \vert \frac{N(\text{Pos},w)}{N(\text{Pos})} - \frac{N(\text{Neg},w)}{N(\text{Neg})}\vert
$$


이 수치를 변형한 것이 네 번째 수치인 확률 비율(Probability Ratio)이다. 정확도 비율에서 - 로 나타나는 부분을 / 로 치환했으며 절댓값을 제거한 수치이다.


$$
\text{PR}(w) = \frac{N(\text{Pos},w)}{N(\text{Pos})} / \frac{N(\text{Neg},w)}{N(\text{Neg})}
$$


다섯 번째 수치는 오즈비(Odds Ratio, OddR)이다. Odds는 성공할 확률을 성공하지 못할 확률로 나눈 것으로 정의된다. 아래 오즈비를 구하는 수식에서는 $w$ 가 등장했을 때의 Class의 문서 수를 등장하지 않았을 때의 숫자로 나누어주어 오즈를 구하였다. Positive Odds를 Negative Odds로 나누어 줌으로써 오즈비를 구하게 된다.


$$
\text{OddR}(w) = \frac{N(\text{Pos},w)}{N(\text{Pos},\bar{w})} \times \frac{N(\text{Neg},\bar{w})}{N(\text{Neg},w)}
$$


오즈비 분자(Odds Ratio Numerator, OddN)는 위 오즈비 식에서 분자 부분만을 남긴 수치이다. 수식도 마찬가지로 오즈비에서 분모를 삭제하여 나타내면 된다.


$$
\text{OddN}(w) = N(\text{Pos},w) \times N(\text{Neg},\bar{w})
$$


F1-Measure는 Confusion matrix에서 Recall과 Precision을 각각 계산한 뒤 그 조화평균을 나타낸 것이다. 수식으로는 아래와 같이 나타낼 수 있다.


$$
\text{F1}(w) = \frac{2 \times N(\text{Pos}, w)}{N(\text{Pos}) + N(w)}
$$


오즈비, 오즈비 분자와 F1은 Negative를 결정하는 Term의 중요성이 상대적으로 무시되는 비대칭적인(Asymmetric) 수치이다. 1~4번째 수치보다는 정확하게 특성의 중요도를 짚어낼 지 모르지만 차원 축소를 위한 절대적인 판단 기준으로 세우기에는 부족한 면이 많다. 앞으로 나올 3개의 수치는 Negative한 Class를 결정하는 특성까지 모두 가중치를 줄 수 있는 대칭적인(Symmetric) 수치이다.

여덟 번째 수치인 정보 획득량(Information Gain, IG)은 대칭적인(Symmetric) 수치 중 하나다. 정보 획득량은 $w$ 의 등장 유무가 드러나지 않았을 때의 엔트로피(Entropy)에서 $w$ 에 의해서 나누어진 클래스의 엔트로피 합을 빼주어 구할 수 있다. 아래는 각각의 엔트로피를 구하는 수식과 정보 획득량을 구하는 수식이다.


$$
\text{Entropy}(\text{absent }w) = \sum_{c \in \{\text{Pos}, \text{Neg}\}} -P(C) \times \log(P(C)) \\
\text{Entropy}(\text{given }w) = P(w)[\sum_{c \in \{\text{Pos}, \text{Neg}\}} -P(C|w) \times \log(P(C|w))] \qquad \\ \qquad \qquad \qquad
+ P(\bar{w})[\sum_{c \in \{\text{Pos}, \text{Neg}\}} -P(C|\bar{w}) \times \log(P(C|\bar{w}))] \\
IG(w) = \text{Entropy}(\text{absent }w) - \text{Entropy}(\text{given }w)
$$


아홉 번째 수치는 카이제곱 통계량(Chi-squared statistic, $\chi^2$ )이다. 해당 변수가 독립일 경우를 가정하고 실제 결과값과의 차이를 구하여 수치로 나타낸다. 카이제곱 통계량을 구하는 수식을 정리하면 아래와 같은 식이 도출된다.


$$
\chi^2(w) = \frac{N \times [P(\text{Pos},w) \times P(\text{Neg}, \bar{w}) - P(\text{Neg},w) \times P(\text{Pos},\bar{w})]^2}{P(w) \times P(\bar{w}) \times P(\text{Pos}) \times P(\text{Neg})}
$$

마지막으로 바이노말 분리(Bi-Normal Separation, BNS)는 각 범주에 대하여 $w$ 가 있을 때와 없을 때의 확률비를 구하고 F통계량(누적확률분포)에 대한 역함수 값의 차이를 구한다. 아래 식을 통해 바이노말 분리를 구할 수 있다.
$$
BNS(w) = \vert F^{-1}(\frac{N(\text{Pos}, w)}{N(\text{Pos})}) - F^{-1}(\frac{N(\text{Neg}, w)}{N(\text{Neg})})\vert
$$


최근에는 분산 표상 방식(Distributed Representation)을 많이 사용하기 때문에 위 방법을 잘 사용하지는 않는다. 굳이 사용한다면 정보 획득량(IG)이나 카이제곱 통계량을 척도로 판별하는 것이 좋다.
