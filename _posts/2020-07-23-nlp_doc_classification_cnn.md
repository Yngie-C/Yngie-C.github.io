---
layout: post
title: 합성곱 신경망(CNN)을 활용한 문서 분류
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# CNN for Sentence Classification

자연어는 시퀀스 데이터이므로 일반적으로 합성곱 신경망(Convolutional Neural Network, CNN)보다는 순환 신경망(Recurrent Neural Network, RNN)을 사용합니다. 시퀀스 데이터만이 가지고 있는 순서를 잘 살릴 수 있는 신경망 구조이기 때문이지요. 하지만 자연어 데이터에서 합성곱 신경망이 아주 쓰이지 않는 것은 아닙니다. 이번 시간에는 **합성곱 신경망을 사용하여 문서를 분류**하는 방법에 대해서 알아보도록 하겠습니다. 

## CNN for Sentence Classification

2014년, Yoon Kim은 [*Convolutional Neural Networks for Sentence Classification*](https://arxiv.org/abs/1408.5882) 논문을 통해 합성곱 신경망을 사용하여 문서를 분류하는 방법을 제안하였습니다. 논문에서 발표한 모델의 구조는 다음과 같습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207983-84473100-cc8b-11ea-8293-4ac74d4c3fb1.png" alt="cnn1" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://arxiv.org/abs/1408.5882">Convolutional Neural Networks for Sentence Classification</a></p>

위 모델의 입력 데이터는 문장을 행렬로 표현한 $M_{n \times k}$ 입니다. 이 행렬에서 각 행은 문장 내 단어를 $k$ 차원의 벡터로 임베딩한 것입니다. 이를 $n$ 개씩 이어붙여 입력 행렬을 만듭니다. 문장 내에 있는 단어의 수가 $n$ 보다 작을 경우에는 패딩(Zero-padding)을 넣어주며, 클 경우에는 길이에 맞춰 자르도록 합니다.



### Vector

위 모델에 사용되는 임베딩 벡터는 **Pre-trained 단어 임베딩 벡터**를 사용합니다. 합성곱 신경망을 통해 학습하는 과정에서 임베딩 벡터가 고정된 값을 가지도록 하는 **Static(정적)** 학습 방식과 학습하면서 값이 변하는 **Non-static(동적)** 방식이 있습니다. 또 여러 형태의 임베딩 벡터를 혼합하여 사용하는 Multi-channel 방법을 적용할 수도 있습니다. 이미지를 합성곱 신경망을 통해서 학습할 때, RGB 값을 각각 처리하기 위해서 3개의 채널을 사용하는 것과 동일한 구조를 가지고 있습니다. 예를 들어, 문서 분류를 위해 3개의 채널을 구성한다면 Word2Vec(Static), GloVe(Static), Word2Vec(Non-static) 처럼 구성할 수 있습니다.



### 1-D Conv

합성곱 신경망을 통해서 이미지를 처리할 때에는 주로 2차원 합성곱(2-D Conv)을 적용합니다. 학습의 대상이 되는 각 픽셀이 평면상에 있는 위치 정보를 가지고 있기 때문입니다. 그래서 필터가 가로 방향으로 슬라이딩하며 합성곱 연산을 수행하면서 가로 방향에 대한 위치정보를 학습할 수 있게 되는 것이지요.

하지만 문서 분류에 합성곱 신경망을 적용할 때 문장을 나타내는 입력 행렬에서 가로 방향을 단어 하나를 임베딩한 벡터입니다. 다시 말해 특정 행 내에서는 요소간에 위치적인 차이가 없다는 것이지요. 이 때문에 문장을 처리하는 합성곱 신경망에서는 세로 방향으로만 슬라이딩하는 **1차원 합성곱(1-D Conv)**이 적용됩니다. 입력 행렬의 크기가 $n \times k$ 이므로 Window는 가로의 길이가 $k$ 인 직사각형입니다. 세로 크기는 조정할 수 있으며 이 값이 커질수록 더 많은 단어를 고려할 수 있게 됩니다. 스트라이드(Stride)는 주로 1을 사용합니다.



### Pooling

합성곱 과정이 끝날 때마다 각 벡터를 하나의 스칼라 값으로 바꾸기 위한 **풀링(Pooling)**을 수행합니다. 논문에서는 최댓값 풀링(Max Pooling)을 사용했을 때가 평균값 풀링(Average Pooling)을 사용했을 때보다 더 좋은 결과를 보여줌을 말하고 있습니다. 해당 문서에서 중점적으로 보아야 하는 부분에 집중하는 것이 모든 단어를 고려하는 것보다 더 좋다고 할 수 있겠습니다. 풀링이 끝나면 각 클래스로 분류를 수행하기 위한 완전 연결 층(Fully-Connected Layer)으로 연산을 이어나가게 됩니다.



### Training

논문에서는 성능을 높이기 위해서 Window의 세로 크기가 $3,4,5$ 인 특성 맵을 각각 100개씩 사용하였습니다. 과적합 방지를 위해서 드롭아웃(Dropout)을 마지막 완전 연결층에 $0.5$ 로 적용하고, 가중치에는 L2 정규화(Regularization)을 적용하였습니다. 미니 배치 사이즈는 $50$으로 하였습니다. 위 값은 SST-2에서 그리드 서치(Grid Search)를 적용하여 찾아낸 것입니다.



### Evaluation

아래는 위 모델을 각각의 데이터셋에 적용한 후에 이전 모델과의 성능을 비교한 표입니다. 우선 위 모델에 4가지 학습 방법을 적용했을 때의 성능을 비교하고 있습니다. CNN-rand는 임베딩 벡터에 랜덤한 값을 부여한 뒤에 학습 과정에서 학습시킨 모델입니다. CNN-static과 CNN-nonstatic은 Pre-trained 단어 임베딩 벡터를 사용하되 학습 과정에서 변하지 않도록 할 지, 변하도록 할 지를 다르게 설정한 모델입니다. CNN-multichannel은 여러 Pre-trained 단어 임베딩 벡터를 설정하여 학습한 것입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207987-8610f480-cc8b-11ea-85e1-4f3cf5b42884.png" alt="cnn2" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://arxiv.org/abs/1408.5882">Convolutional Neural Networks for Sentence Classification</a></p>

결과에서 주목할 점은 static이든 non-static이든 multichannel이든 Pre-trained 임베딩 벡터를 사용하는 것이 랜덤 임베딩 벡터로 학습하는 것보다 결과가 좋다는 것입니다. 여러 가지 임베딩 벡터를 혼합한 multichannel의 성능이 가장 높을 것으로 예측해볼 수 있지만 실제로는 데이터셋마다 최고 성능을 보이는 모델이 다른 것을 알 수 있습니다. 기존 모델과 비교해 보았을 때에는 7개의 데이터셋 중 4개의 데이터셋에서 합성곱 신경망을 사용한 모델이 가장 좋은 성능을 보였음을 알 수 있습니다.



## Character-Level CNN

1년이 지난 2015년에는 [*Character-level Convolutional Networks for Text Classification*](https://papers.nips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf) 논문을 통해 Character 단위에서 합성곱을 수행하는 Character-Level CNN이 고안되었습니다. 전체적인 모델의 구조는 아래와 같습니다.



<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207958-7c878c80-cc8b-11ea-8397-0cb68e1e7fac.png" alt="cnn3" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://papers.nips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf">Character-level Convolutional Networks for Text Classification</a></p>

위 모델에서는 총 70개의 Character를 수량화하여 나타내며 이는 26개의 알파벳과 10개의 숫자, 33개의 특수문자와 1개의 공백 문자로 구성되어 있습니다.

### Vector

Character-Level CNN에서 문장이 행렬로 표현되는 방식은 문장을 단어 단위 벡터의 행렬로 나타내었던 방식과 다릅니다. 후자가 바로 임베딩 벡터를 적용하였다면, 전자는 우선 각 Character를 70차원의 원-핫 인코딩 벡터(One-hot Encoding Vector)로 표현한 뒤에



Large Feature의 경우 1024 길이를 사용하므로 행렬의 크기는 70*1024가 된다. 



### Conv & Pooling

이 모델에는 총 6개의 합성곱 층이 사용되며 앞단에 있는 4개의 합성곱 층에는 Window 사이즈가 7인 커널을 사용하고 나머지 2개 층에서는 Window 사이즈가 3인 커널을 사용합니다. 그리고 합성곱 층 다음마다 풀링을 모두 적용하는 것이 아니라 적용하는 층이 있고 그렇지 않은 층이 있습니다. 풀링을 적용하는 층에는 Window 사이즈가 3인 최댓값 풀링을 사용합니다. 6개의 층을 모두 지난 이후에는 2개의 완전 연결층으로 넘어가게 됩니다.



### Data Augmentation

논문에서는 데이터 증식(Data augmentation)에 대한 방법도 제시하고 있습니다. 이미지를 처리할 때에는 이를 회전시키거나 좌우 혹은 상하 반전시켜 데이터를 증식하는 등 여러 방법을 적용할 수 있습니다. 하지만 텍스트의 경우 이런 방법을 적용할 수 없기 때문에 새로운 방식을 사용합니다. 논문에서는 동의어(Synonym)로 치환하여 데이터를 늘리는 방법을 사용하였습니다.



### Evaluation

아래는 문서 분류에 대한 Character-Level CNN과 기본 모델의 성능을 비교한 표입니다. 

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207963-7d202300-cc8b-11ea-9fce-fd7c7a61ae0c.png" alt="cnn4" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://papers.nips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf">Character-level Convolutional Networks for Text Classification</a></p>

크기가 작은 데이터셋에서는 TF-IDF가 가장 좋은 성능을 보이며 크기가 일정 이상으로 커졌을 때에는 합성곱 신경망을 사용한 모델의 성능이 확연히 높아지는 것을 볼 수 있습니다.



## Addition

### How deep is it?

2017년에는 합성곱 신경망의 깊이(Depth)가 문장 분류에 있어서 어떤 영향을 끼치는 지를 연구한 논문인 [*Do Convolutional Networks need to be Deep for Text Classification?*](https://arxiv.org/abs/1707.04108) 이 발표되었습니다. 논문에서는 아래 그림과 같은 2개의 합성곱 신경망 모델을 비교하고 있습니다. 왼쪽은 첫 번째로 알아본 것과 같이 합성곱 층이 얇은 모델을 나타냅니다. 오른쪽은 2016년 [*Densely Connected Convolutional Networks*](https://arxiv.org/abs/1608.06993) 논문에서 발표된 DenseNet과 같이 합성곱 층을 깊게 쌓은 모델을 나타냅니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207967-7e515000-cc8b-11ea-973d-90c37b17d6ef.png" alt="cnn5" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">github.com/pilsung-kang</a></p>

논문에서는 이 둘을 비교하며 Character-Level 에서는 합성곱 층을 깊게하는 것이 성능이 좋고, 단어 단위로 벡터를 구성할 때에는 합성곱 층을 많이 쌓지 않는 것이 좋다고 발표하였습니다.

### Localization

18년도에는 말뭉치의 감성을 분석한 뒤 CAM(Class Activation Map)을 활용하여 시각화한 논문이 발표되었습니다. 해당 논문에서는 단어 단위의 CNN을 사용하였습니다. 문장 앞뒤로 패딩을 준 뒤에 Window 사이즈를 $3,4,5$ 로 다르게 설정하고 에버리지 풀링(Average pooling)을 적용하였습니다. 학습 후 얻어낸 가중치를 통해, 어떤 단어를 보고 긍/부정을 판별하는 지를 시각화하였습니다. 아래는 해당 논문에서 IMDB 데이터셋을 시각화한 이미지입니다. 위는 Positive로 분류된 문서이며, 아래는 Negative로 분류된 문서입니다. 이렇게 분류되는데에 어떤 단어가 결정적인 영향을 미쳤는지에 대해서 잘 보여주고 있습니다.



<p align="center"><img src="https://user-images.githubusercontent.com/45377884/88207970-7f827d00-cc8b-11ea-8a4f-1f3e2cbc0a85.png" alt="cnn7" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">github.com/pilsung-kang</a></p>


