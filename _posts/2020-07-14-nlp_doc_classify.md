---
layout: post
title: 문서 분류 - Vector Space Model
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# 문서 분류

**문서 분류(Document Classification/Categorization)** 는 감성분석, 스팸 분류, 저자별 분류 등 다양한 Task에 사용되는 기술이다.

문서 분류는 대표적인 **지도학습(Supervised Learning)** 기반의 머신러닝 기법이다. 학습 데이터셋(Train set)의 특성(Feature) $X$ 으로는 각각의 문서가 들어가며 레이블(Label) $Y$ 로는 각 문서가 속한 범주(Category)가 입력된다. 그리고 테스트 데이터셋(Test set)에서 주어진 특성을 기준으로 레이블을 판별하는 정도를 평가하게 된다.

이를 학습하기 위한 알고리즘은 여러가지가 있다. 아래를 보면 어떤 문서 집합에 대해서 문서를 이진 분류(Binary Classification)했을 때의 결정 경계를 나타낸 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/87411487-c248b300-c602-11ea-8ca0-93b12b7db804.png" alt="nlp_dc" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

이렇게 많은 알고리즘이 존재하는 이유는 모든 데이터셋에서 최고의 성능을 보이는 알고리즘이 없기 때문이다. 데이터셋이 달라짐에 따라 성능이 높은 학습 알고리즘이 다르므로 여러 알고리즘을 알고 있어야 한다.



## 벡터 스페이스 모델

**벡터 공간 모델(Vector Space Model)** 이란 하나의 문서를 DTM(Document-Term Matrix)을 사용하여 하나의 벡터로 표현한 것이다.

![nlp_dtm](https://user-images.githubusercontent.com/45377884/87413116-d8f00980-c604-11ea-8b43-1578b950ff67.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

## Naive Bayes Classifier

문서 분류를 위한 여러가지 알고리즘 중 하나인 **나이브 베이즈 분류기(Naive Bayes Classifier)** 에 대해 알아보자. 나이브 베이즈 분류기는 베이즈 규칙(Bayes' Rule)에 가정(Naive Assumption)을 더한 것이다. 먼저 베이즈 규칙에 대해 알아보자. 베이즈 규칙을 수식으로 나타내면 다음과 같다.


$$
P(C_i|x_1, x_2) = \frac{P(x_1, x_2|C_i) \cdot P(C_i)}{P(x_1, x_2)}
$$


특성 $x_1, x_2$ 가 주어졌을 때 문서의 범주가 $C_i$ 일 확률을 다음과 같이 나타낼 수 있다. 여기에 각각의 특성(단어)이 독립이라는 가정을 더하면 결합 확률을 하나씩 나누어 나타낼 수 있다.


$$
P(C_i|x_1, x_2) = \frac{\color{blue}P(x_1, x_2|C_i) \cdot \color{red}P(C_i)}{P(x_1, x_2)} = \frac{\color{blue}P(x_1|C_i) \cdot P(x_2|C_i) \cdot \color{red}P(C_i)}{P(x_1, x_2)}
$$


위 수식에서 빨간색으로 표시된 부분은 사전확률(Priority probability)이며 파란색으로 표시된 부분은 클래스가 주어졌을 때 각 특성이 등장할 조건부 확률이다. 각 부분을 구하는 방법은 다음과 같다.


$$
{\color{red}\hat{P}(c_j)} = \frac{N.Doc(C = c_j)}{\text{Total number of documents}} \\
{\color{blue}\hat{P}(w_i|c_j)} = \frac{\text{count}(w_i,c_j)}{\sum_{w \in V} \text{count}(w,c_j)} \\
$$


사전 확률은 특정 클래스에 속하는 문서의 개수를 전체 문서의 개수로 나누어 구할 수 있고. 조건부 확률은 특정 클래스의 문서에서 특정 단어가 등장한 횟수를 모든 단어가 등장한 횟수로 나누어 구할 수 있다.

이 방법은 한 가지 단점을 가지고 있다. 특정 클래스의 문서에 한 번도 등장하지 않은 단어가 있을 경우 그 단어의 조건부 확률은 0이 된다. 예를 들어 *"negative"* 클래스로 분류된 모든 문서에 *"fantastic"* 이라는 단어가 하나도 포함되어 있지 않을 경우를 생각해보자. 이 때의 조건부 확률은 아래와 같이 나타난다.


$$
\text{count}({\color{ProcessBlue}\text{fantastic}},{\color{RedOrange}\text{negative}}) = 0 \quad 이면 \\
\hat{P}({\color{ProcessBlue}\text{fantastic}}|{\color{RedOrange}\text{negative}}) = 0 \quad 이므로 \\
\hat{P}({\color{RedOrange}\text{negative}}|\cdots, {\color{ProcessBlue}\text{fantastic}}, \cdots) = 0
$$


이러한 문제를 해결하기 위해 스무딩(Smoothing)이라는 기법이 고안되었다. 가장 기본적인 스무딩방법인 라플라스 스무딩의 경우 모든 단어의 발생 횟수에 1을 더해준다. 이러한 방법을 통해 특정 클래스에 한 번도 등장하지 않는 단어가 있더라도 확률이 0이 되는 것을 막는다. 라플라스 스무딩을 수식으로 나타내면 아래와 같다.


$$
\hat{P}(w_i|c_j) = \frac{\text{count}(w_i,c_j)+1}{\sum_{w \in V} (\text{count}(w,c_j)+1)} = \frac{\text{count}(w_i,c_j)+1}{\sum_{w \in V} \text{count}(w,c_j)+V}
$$


## kNN Classifier

이번에는 문서 분류를 위한 간단한 알고리즘 중 **k-Nearest Neighbor Classifier(kNN Classifier)** 에 대해 알아보자. kNN 알고리즘의 기본적인 아이디어는 '유유상종'이라는 사자성어에 근거한다. 우리가 분류해야 할 문서와 가장 유사한 k개의 문서가 어떤 클래스인지를 보고 가장 유사한 클래스로 분류하게 된다. 아래의 사진은 $k=3$ 일 때 문서를 분류하는 방법을 나타낸 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/87423325-13fa3900-c615-11ea-9c82-f5e1d25a94cd.png" alt="nlp_knn" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

kNN 알고리즘을 적용하는 방법은 다음과 같다. 먼저, 레퍼런스(위 그림에서 파란색 원, 주황색 사각형에 해당하는) 데이터를 준비한다. 두 번째로, 유사도에 대한 수치를 정의한다. 유사도는 거리에 반비례 하므로 어떤 거리(Norm)를 사용할 것인지를 결정하는 단계가 된다. 일반적으로는 L2 norm인 Euclidean Norm을 많이 사용하며 경우에 따라 L1 Norm등 다양한 거리를 사용하기도 한다. 세 번째는 몇 개의 레퍼런스를 찾을 것인지에 대한 k 를 설정한다. k 가 너무 작을 경우 지역 특성에 너무 민감하게 반응하여 과적합되는 경향이 있고, 반대로 k 가 너무 크면 지역 특성을 무시해버려 과소적합되는 경향을 보인다. 아래는 임의의 데이터셋에 대해 각각 k가 1, 10, 20, 50일 때에 대한 결정 경계를 나타낸 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/87424639-43aa4080-c617-11ea-96a3-8a952d88d993.png" alt="nlp_knn_k" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

마지막으로 레이블을 판단하기 위한 기준을 정한다. 기준은 다수결 방법(Majority Voting)과 가중치 방법(Weighted Voting)이 있다. 다수결 방법은 선별한 k개의 문서 중에 어떤 클래스의 문서가 더 많은지를 보고 해당 문서의 클래스를 정한다. 가중치 방법은 가장 유사도가 높은(거리가 가까운) 레퍼런스에 가중치를 부여한다. 레퍼런스와의 유사도를 총 합이 1이 되도록 바꿔준 후 각 클래스 끼리 더하여 값이 더 높은 클래스로 분류하게 된다. 아래 이미지는 새로운 문서 X를 k=5일 때 다수결 방법과 가중치 방법을 활용하여 나누어준 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/87426550-4b1f1900-c61a-11ea-97c2-f53b43a88dde.png" alt="nlp_knn_vote" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

위 그림에서 다수결 방법을 사용할 경우에는 5개의 레퍼런스 중 *'JoF'* 로 분류된 것이 더 많으므로 새로운 데이터는 *'JoF'* 로 분류된다. 하지만 가중치 방법을 사용할 경우에는 각 클래스의 가중치가 *'TPAMI(0.59), JoF(0.41)'* 로 값이 더 높은 *TPAMI* 로 분류된다. 

kNN 알고리즘을 사용할 때에도 몇 가지 신경써야할 사항이 있다. 첫 번째는 Normalization이다. 각 특성이 단어인 경우에는 크게 상관이 없지만, 키-몸무게-성별 등 다양한 단위를 가지는 경우에는 정규화(Normalization)를 해주어야 거리를 올바르게 잴 수 있다.

두 번째로는 한 쪽으로 치우쳐진(Stratified) 데이터셋에 대해 컷오프(Cut-off)를 적용하는 방식이다. 클래스 A가 100명, 클래스 B가 400명으로 이루어진 전체 500명에 대한 데이터셋이 있다고 해보자. 새로운 데이터에 대해 다수결 방법을 사용하여 클래스 A일 확률이 $P(X=A) = 0.4$ 가 나왔다고 한다. 컷오프를 사용하지 않았다면 $P(X=B) = 0.6$ 이므로 새로운 데이터는 B클래스로 분류될 것이다. 하지만, 위와 같이 데이터가 한 쪽으로 치우쳐져 있을 때는 각 클래스에 대한 기댓값을 구하여 Threshold를 조정해준다. 해당 데이터셋에서 A 클래스의 기댓값이 $E(A) = 0.2$ 이고 $P(X=A) = 0.4$ 가 기댓값보다 높으므로 A클래스로 분류하게 된다.