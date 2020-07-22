---
layout: post
title: CNN을 활용한 문서 분류
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# CNN for Sentence Classification

2014년에는 문장(혹은 문서)의 분류를 위해서 합성곱 신경망(Convolutional Neural Network, CNN)을 사용하는 방법이 고안되었다. 논문에서 발표한 모델의 구조는 아래와 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207983-84473100-cc8b-11ea-8293-4ac74d4c3fb1.png" alt="cnn1" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

위 신경망에 들어가는 입력 데이터는 문장을 행렬로 표현한 $M_{n \times k}$ 이다. 각 행은 하나의 단어를 $k$ 차원의 임베딩 벡터로 나타낸 것이며 이를 $n$ 개씩 이어붙여 입력 행렬을 구성한다. 문장 내에 있는 단어의 수가 $n$ 보다 작을 경우에는 패딩(zero-padding)을 넣어주며, $n$ 보다 클 경우에는 길이에 맞춰 자르게 된다. 이 때 사용하는 임베딩 벡터는 Pre-trained 단어 임베딩 벡터를 사용한다. 이렇게 가져와진 임베딩 벡터에 학습 과정을 거쳐도 변하지 않는 static 방법을 적용할 수도 있고, 학습 과정에서 같이 학습되는 non-static 방법을 적용할 수도 있다. 또한, 하나의 방법만을 사용하는 것이 아니라 여러 개의 임베딩을 혼합하여 사용하는 Multi-channel 방법을 선택할 수도 있다. 이는 이미지에서 CNN을 할 때, RGB값을 각각 처리하기 위해서 3개의 채널을 사용하는 것과 비슷하다. 만약 3개의 채널을 구성한다면 Word2Vec(Static), GloVe(Static), Word2Vec(Non-static) 으로 구성할 수 있으며 이외에 다른 임베딩 벡터들도 자유롭게 가져다 쓸 수 있다.

이미지를 처리하는 CNN은 주로 2차원 합성곱(2D-Conv)을 적용한다. 각각의 픽셀이 위치정보를 가지고 있기 때문이다. 그래서 필터가 가로 방향으로 슬라이딩하며 합성곱을 수행하는 것이 의미를 가진다. 하지만 문장을 구성하는 행렬에서 가로 방향은 각 단어 하나를 임베딩한 벡터이다. 즉, 특정 행에서의 첫 번째 요소와 마지막 요소 간에 위치적인 차이가 없다. 그렇기 때문에 문장을 처리하는 CNN에는 세로 방향으로만 합성곱을 수행하는 1차원 합성곱(1D-Conv)이 적용된다. Window는 가로의 길이가 k인 직사각형이며 세로 크기는 달라질 수 있다. 스트라이드(Stride)는 주로 1을 사용하며, Window의 세로 크기가 커질수록 더 많은 단어를 고려하게 된다.

합성곱 과정이 끝나면 각 벡터를 하나의 스칼라 값으로 바꾸기 위한 풀링(Pooling)을 수행하게 된다. 논문에서 최대 풀링(Max Pooling)을 사용하여 해당 문서에서 중점적으로 보아야 하는 파트에 집중하는 것이 평균 풀링(Average Pooling)을 사용하여 모든 단어를 고려하는 것보다 더 좋은 결과를 보여주었다. 그래서 이 논문에서는 Max Pooling 을 사용하고 있다.  풀링 과정이 끝나면 각 클래스로의 분류를 수행하기 위한 완전 연결 층(Fully-Connected Layer)으로 넘어가게 된다.

논문에서는 결과를 높이기 위해서 다음과 같은 학습 전략을 선택하였다. Window의 세로 크기가 각각 3, 4, 5인 각각의 특성 맵 100개 씩(총 300개)을 사용하였다. 과적합 방지를 위한 드롭아웃(Dropout)은 마지막 완전 연결층에만 0.5의 값을 적용하였으며 가중치에 L2 규제를 적용하였다. 미니 배치 사이즈는 50개를 사용하였다. 이 값들은 SST-2에서 그리드 서치(Grid Search)를 적용하여 찾아낸 것이다.

아래는 이 모델을 각각의 데이터셋에 적용한 후 다른 많은 모델과의 성능을 비교한 것이다. 해당 모델 또한 4가지 경우를 나누어 성능을 비교하고 있다. CNN-rand는 초기 임베딩 값으로 랜덤한 값을 주고 학습 과정에서 학습시킨 모델이다. CNN-static과 CNN-nonstatic은 앞서 말한 것과 같이 Pre-trained 단어 임베딩 벡터를 사용한 뒤 학습과정에서 불변하는 값으로 두거나, 변하는 값으로 둔 모델이다. CNN-multichannel은 여러 개의 Pre-trained 임베딩 벡터를 학습한 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207987-8610f480-cc8b-11ea-85e1-4f3cf5b42884.png" alt="cnn2" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

위 결과에서 주목할 점은 static이든 non-static이든 multichannel이든 Pre-trained 임베딩 벡터를 사용하는 것이 랜덤으로 학습하는 것보다 결과가 좋다는 것이다. 통념상 셋 중에서는 여러가지 임베딩 벡터를 혼합한 multichannel의 성능이 가장 높을 것 같지만 각 데이터셋 마다 최고 성능 모델이 다른 것을 알 수 있다.



## Character-Level CNN

2015년에는 Character 단위에서 CNN을 수행하는 Character-Level CNN이 고안되었다. 총 70개의 Character를 수량화하여 나타내며 이는 26개의 알파벳, 10개의 숫자, 33개의 특수문자와 1개의 공백문자로 구성되어 있다.

Character-Level CNN에서 문장이 행렬로 표현되는 방식은 단어 단위에서 문장이 행렬로 변환했던 것과 다르다. 먼저, 각 Character를 70차원의 원-핫 인코딩 벡터(One-hot Encoding Vector)로 표현한다. Large Feature의 경우 1024 길이를 사용하므로 행렬의 크기는 70*1024가 된다. 이 모델에는 총 6개의 합성곱 층이 사용되며 앞단에 있는 4개의 층은 Window 사이즈가 7인 커널을 사용하며 그 다음 2개의 층에서는 사이즈가 3인 커널을 사용한다. 6개의 합성곱 층에는 풀링을 적용하는 층도 있고 그렇지 않은 층도 있다. 풀링을 적용하는 층에서는 사이즈가 3인 Max Pooling이 사용된다. 합성곱층을 모두 지난 이후에는 2개의 완전 연결층으로 넘어간다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207958-7c878c80-cc8b-11ea-8397-0cb68e1e7fac.png" alt="cnn3" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

해당 논문에서는 CNN에서 데이터 증식(Data Augmentation)하는 방법을 제시하고 있다. 이미지의 경우 회전, 반전 등 데이터를 증식하는 방법이 다양하다. 하지만 문서의 경우 이런 방법을 사용할 수 없으므로 유의어(Synonym)를 치환하여 데이터를 늘리는 방법을 사용한다.

아래는 Character-Level CNN과 TF-IDF 등 기존 모델의 문서 분류 성능을 비교한 표이다. 데이터셋의 크기가 작은 경우에는 TF-IDF가 가장 좋은 성능을 보이며, 크기가 일정 이상으로 커지면 CNN이 최고 성능을 보이는 것을 알 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207963-7d202300-cc8b-11ea-9fce-fd7c7a61ae0c.png" alt="cnn4" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

## 추가 연구들

2018년에 발표된 논문에서는 CNN층의 깊이(Depth)가 문장 분류에 있어서 얼마만큼의 영향을 미치는 지를 연구했다. 이 논문에 의하면 Character-Level 에서는 합성곱 층을 깊게하는 것(아래 그림 오른쪽)이 성능이 좋고, 단어 단위에서는 합성곱 층을 많이 쌓지 않는 것이 성능이 좋다고 발표했다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207967-7e515000-cc8b-11ea-973d-90c37b17d6ef.png" alt="cnn5" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

2018년에는 감성분석 Task에 CAM(Class Activation Map)을 활용한 시각화(Localization)에 대한 연구가 발표되었다. 이 논문에서는 단어 단위의 CNN을 사용하였으며 문장의 앞뒤로 패딩을 준 뒤 Window 사이즈를 3,4,5로 다르게 하여 평균 풀링(Average Pooling)을 적용하였다. 그리고 최종적인 가중치가 주어지면 긍정과 부정을 판별하기 위해서 어떤 단어를 중요하게 살피는 지를 CAM으로 표현하였다. 아래는 해당 논문에 있는 IMDB 데이터셋을 Localization한 결과이다. 각각 Positive와 Negative로 분류되는 데에 결정적인 영향을 미친 단어들을 시각화하여 보여주고 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207970-7f827d00-cc8b-11ea-8a4f-1f3e2cbc0a85.png" alt="cnn7" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>


