---
layout: post
title: 트랜스포머 (Transformer)
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# Transformer

## Attention is All You Need

2013년 Word2Vec이 발표된 후에 GloVe, Fasttext 등이 등장하면서 단어 임베딩에 대한 방법론이 어느 정도 마무리 되었다. 그 뒤를 이어 문장 임베딩을 위한 기술들이 등장하게 되었고 2017년도 6월에 *"Attention is All You Need"(Ashish Vaswani et al.)* 라는 논문에 등장한 것이 **트랜스포머(Transformer)** 이다. 트랜스포머는 어텐션(Attention) 메커니즘을 극대화하기 위한 방법이다. 시퀀스를 입력받아 시퀀스를 내뱉는다는 점에서는 RNN과 유사하다. 하지만 내부에서는 시퀀스가 RNN처럼 순차적으로 처리되지 않고 한꺼번에 계산된다. 그렇기 때문에 학습 과정에서 병렬화(Parallelize)가 용이하여 속도를 높일 수 있다.    

![transformer_1](http://jalammar.github.io/images/t/the_transformer_3.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

트랜스포머의는 이전에 배운 모델과 유사한 인코더(Encoder)-디코더(Decoder) 구조를 가지고 있다. 하지만 둘을 구성하고 연결하는 방법, 그리고 내부에서 일어나는 연산과정에서 이전 모델과의 차이점을 가진다.

<p align="center"><img src="http://jalammar.github.io/images/t/The_transformer_encoders_decoders.png" alt="transformer_2" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

## 트랜스포머의 내부구조

이전의 모델은 내부에 인코더-디코더 하나씩을 가지고 있었다. 하지만 트랜스포머는 인코더와 디코더를 각각 여러 개 중첩(Stack)한 구조를 갖는다. 발표된 논문에서는 6개씩의 인코더와 디코더를 쌓았다. (하지만, 6개라는 숫자가 특별한 의미를 가지는 것은 아니다) 아래 이미지는 논문에서 발표한 트랜스포머의 구조를 인코더와 디코더 개수까지 도식화하여 나타낸 것이다.

<p align="center"><img src="http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png" alt="transformer_3" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

이렇게 구성된 각 인코더 블록의 구조는 모두 동일하다. 하지만 그들이 서로 가중치를 공유하거나 하지는 않는다. 인코더의 블록의 구조부터 자세히 살펴보자. 인코더 블록은 2개의 하위 레이어(Sub-Layer)로 구성되어 있다. 첫 번째는 셀프 어텐션(Self-Attention) 층이며 두 번째는 FFNN(Feed Forward Neural Network) 층이다. Self-Attention은 한 토큰에 대한 정보를 처리할 때 이를 처리하기 위해서 입력 시퀀스의 다른 토큰을 얼마나 중요하게 생각할 것인가를 계산한다. Self-Attention 으로부터 계산된 아웃풋은 FFNN으로 들어가게 된다. FFNN에서는 이를 활용하여 가장 적절한 출력을 산출하게 된다. 

<p align="center"><img src="http://jalammar.github.io/images/t/Transformer_encoder.png" alt="transformer_4" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

디코더는 인코더보다 하나의 하위 레이어를 더 가지고 있다. 가장 아랫부분에 Self-Attention층, 가장 윗부분에 FFNN을 가지는 것은 동일하지만 가운데 하나의 층이 더 들어간다. 이 하위 레이어는 인코더-디코더 어텐션이라고 불리며 인코더가 가지고 온 정보에 대해서 어텐션 메커니즘을 실시한다. 아래 그림은 인코더의 2중 구조, 디코더의 3중 구조를 도식화한 것이다.

![transformer_5](http://jalammar.github.io/images/t/Transformer_decoder.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

## Input Embedding & Positional Encoding

트랜스포머의 전체적인 메커니즘을 도식화하면 다음과 같이 나타낼 수 있다. 아래 이미지에서는 인코더와 디코더 블록 하나씩만을 나타냈으며 실제로는 각 N개의 블록 스택(논문에선 N=6)을 쌓아 구성한다. 아래 그림에서 주목할 것 하나는 Self-Attention이 Multi-Head Attention이라는 것이며 두 번째는 디코더에서는 일부가 마스킹된 Masked Self-Attention을 사용한다는 것이다.

![transformer_6](https://paul-hyun.github.io/assets/2019-12-19/transformer-model-architecture.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://paul-hyun.github.io/transformer-03/">Paul-Hyun Github</a></p>

먼저 가장 첫 과정인 **입력 시퀀스를 임베딩(Input Embedding)** 하는 과정부터 알아보자. 인풋 임베딩은 입력 시퀀스의 아이템을 첫 번째 인코더 블록에 입력할 수 있도록 벡터화하는 것이다. 또한 첫 번째 블록에 입력할 때만 진행한 후 이후로는 진행하지 않는다. 일반적으로는 Word2Vec, GloVe, Fasttext 등의 단어 임베딩 방법을 사용한다. 트랜스포머의 인코더와 디코더는 입출력시 512차원의 벡터를 다루기 때문에 인풋 임베딩을 거친 단어는 512차원의 벡터로 변하게 된다. 하나의 시퀀스로 처리할 리스트의 크기는 따로 설정해주어야 하는 하이퍼파라미터이며 기본적으로는 훈련 데이터셋에 있는 가장 긴 문장의 길이로 사용한다.

그 다음 해주어야 할 작업은 **포지셔널 인코딩(Positional Encoding)** 이다. 트랜스포머는 RNN처럼 토큰을 순차적으로 다루지 않기 때문에 연산 과정에서 단어의 순서 정보를 무시하게 된다. 이를 완벽히 보존하지는 못하더라도 단어의 위치를 어느정도 보완해줄 수 있는 장치가 필요한데 이를 위해 동작하는 것이 포지셔널 인코딩이다. 인풋 임베딩을 거친 벡터에 포지셔널 인코딩된 벡터를 더해주어 단어의 위치정보를 조금이나마 복원한다.

포지셔널 인코딩에 대해 좀 더 자세히 들여다보자. 좋은 포지셔널 인코딩은 다음과 같은 조건을 만족하여야 한다. 먼저 인코딩 벡터의 거리(Norm)는 모든 위치에 대해서 동일하게 주어져야 한다. 위에서 보았듯 포지셔널 인코딩을 거쳐 나온 벡터는 인풋 임베딩 벡터에 더해지게 된다. 그런데 위치마다 벡터의 거리(크기)가 달라지게 되면 같은 방향, 같은 크기로 변한다는 것을 보장할 수 없게 된다. 두 번째는 둘 사이의 거리가 멀어지게 되면 포지셔널 인코딩 벡터 사이의 거리도 멀어져야 한다. 아래는 이러한 조건을 대략적으로 만족시키는 포지셔널 인코딩의 예시를 나타낸 수식이며 표는 생성된 벡터 사이의 거리를 L2 노름(Norm)으로 나타낸 결과를 시각화한 것이다.


$$
PE_{(\text{pos}, 2i)} = \sin(\text{pos}/10000^{2i/d_{\text{model}}}) \\
PE_{(\text{pos}, 2i+1)} = \cos(\text{pos}/10000^{2i/d_{\text{model}}})
$$

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/86228459-5e1cfc80-bbc9-11ea-9443-355859646b0f.png" alt="posi_encode1" style="zoom:80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

위 표에서 수치를 대체로 살펴보면 가까이 있는 단어는 값이 낮고 멀리 있을 수록 수치가 커지는 것을 알 수 있다. 물론 멀리 있는 토큰에 대하여 무조건 숫자가 커지지는 않지만 일반적인 경향성은 만족하면서 대체적인 거리 관계를 반영하고 있음을 알 수 있다.



## Encoder Block

포지셔널 인코딩 벡터가 더해진 인풋 임베딩 벡터는 이제 인코더의 첫 번째 블록에 들어가게 된다. 위에서 설명한 바와 같이 인코더의 각 블록에는 2개의 하위 레이어가 있다. 

<p align="center"><img src="http://jalammar.github.io/images/t/encoder_with_tensors_2.png" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

포지셔널 인코딩까지 마친 토큰 벡터 $x_1, x_2$ 가 인코더 블록에서 가장 먼저 만나는 층은 **셀프 어텐션(Self-Attention)** 이고 이를 거쳐 나온 출력 벡터 $z_1, z_2$ 는 **FFNN(Feed Forward Neural Network)** 층을 지나게 된다. Self-Attention 층에서는 각 토큰 벡터가 서로 영향을 끼치는(Dependency) 관계에 있지만 FFNN에서는 서로 영향을 끼치지 않는다. 그렇기 때문에 FFNN 단계에서는 병렬화가 가능하며 학습 속도를 더욱 빠르게 가져갈 수 있다. 첫 번째 블록의 최종 아웃풋인 $r_1, r_2$ 벡터는 두 번째 인코더 블록의 인풋으로 사용된다.



## Self-Attention

셀프 어텐션의 역할은 무엇일까? 다음과 같은 문장이 있다고 해보자.


$$
\text{"The animal didn't cross the street because it was too tired"}
$$


사람은 위 문장에서의 *"it"* 이 무엇을 가리키는지 쉽게 알 수 있지만 컴퓨터에게는 쉽지만은 않은 문제다. 이렇게 문장에 있는 어떤 단어는 다른 단어에 많은 영향을 끼치고 있다. 그래서 각 단어가 시퀀스 내에 있는 다른 단어에게 어떤 영향을 끼치는지 살펴보겠다는 것이 바로 Self-Attention의 목적이다. 아래는 6개의 인코더 블록 중 5번째 블록에서 *"it"* 이 어떤 단어에 영향을 가장 많이 끼치는 지를 나타낸 그림이다. 

![transformer_8](http://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

Self-Attention 층에서 어떤 계산이 일어나는 지를 좀 더 자세히 알아보자. 문제를 풀기 위해서 세 가지 벡터를 준비해야 한다. 각각의 벡터는 **쿼리(Query, Q), 키(Key, k), 밸류(Value, V)** 라고 부른다. 가장 먼저 쿼리는 현재 프로세싱 하고 있는 토큰을 나타내는(Representation) 벡터이다. 키는 일종의 레이블로서 시퀀스 내 모든 토큰에 대한 아이덴티티를 나타낸다. 키는 프로세싱 중인 토큰의 쿼리 벡터와의 연산을 통해 얼마나 영향을 미치는 지에 대한 Score를 매긴다. 마지막으로 밸류는 키와 연결된 실제 토큰을 나타내는 벡터이다. 아래의 그림을 보자.

<p align="center"><img src="http://jalammar.github.io/images/gpt2/self-attention-example-folders-3.png" alt="transformer_9" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

위의 그림에서 쿼리로 나타나는 Query #9 벡터는 10번째 단어인 "it"을 표현한다. 컴퓨터는 이 쿼리를 보고 모든 키를 살펴본 뒤 가장 잘 맞는 키를 찾는다. 가장 잘 맞는지 확인하는 방법은 벡터의 내적(Dot product)으로 이루어지며 내적을 통해 나온 값은 서로가 얼마나 영향을 끼치고 있는지를 나타내는 점수(Score)가 된다. 마지막으로 이 스코어를 밸류와 곱하여 어떤 단어와 얼마나 영향을 끼치는 지를 구하게 된다.

<p align="center"><img src="http://jalammar.github.io/images/t/transformer_self_attention_vectors.png" alt="transformer_10" style="zoom: 80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

각각의 쿼리, 키, 밸류 벡터는 어떻게 만들어질까? Self-Attention 층 내부에는 세 벡터를 만들기 위한 행렬 $W^Q, W^K, W^V$ 가 미리 준비되어 있다. 각각의 행렬과 인풋 토큰 벡터를 내적하여 인풋 토큰에 맞는 쿼리 - 키 - 밸류 벡터를 만들게 된다. 예를 들어, $x_1 \cdot W^Q = q_1, x_1 \cdot W^K = k_1, x_1 \cdot W^V = v_1$ 가 되며 $x_2 \cdot W^Q = q_2, x_1 \cdot W^K = k_2, x_1 \cdot W^V = v_2$ 가 된다. 여기서 $W$ 행렬에 해당하는 $W^Q, W^K, W^V$ 각 요소의 값은 학습을 통해 찾아야 하는 미지수에 해당한다.

일반적으로 쿼리 - 키 - 밸류 벡터의 차원은 인풋 토큰 벡터의 차원보다 적게 설정한다. (무조건 작게 설정해야 하는 것은 아니다) 논문에서는 인풋 토큰 벡터를 512차원의 벡터로 만들어주며 쿼리, 키, 밸류는 64차원의 벡터로 설정한다. 나중에 Multi-Head Attention을 활용할 때 각 아웃풋을 Concatenate를 해주는데 이를 위해 더 작은 차원으로 설정하게 된다. 만약 인풋이 512차원, 쿼리 - 키 - 밸류가 64차원이라면 512/64, 즉 8은 Multi-Head Attention의 숫자가 된다.

세 가지 벡터(쿼리 - 키 - 밸류)를 만들어 주었다면 그 다음은 아래와 같이 쿼리와 가장 잘 맞는 키를 찾는 과정이다.

<p align="center"><img src="http://jalammar.github.io/images/gpt2/self-attention-example-folders-scores-3.png" alt="transformer_10" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

쿼리와 키가 잘 맞는지는 각 벡터의 내적을 이용하여 그 값을 사용한다. 그 다음으로는 해당 쿼리와 각 키의 내적값을 차원의 제곱근 값으로 나누어준다. 이는 그래디언트를 안정적으로 만들어주기 위함이며 논문에서는 64차원의 벡터를 사용하고 있기 때문에 그 값을 8로 나누어주게 된다. 차원의 제곱근으로 나누어준 값에 소프트맥스를 취해주면 프로세싱 중인 토큰이 각 토큰에 얼마나 영향을 끼치는지에 대한 비율이 나오게 된다. 최종적으로 이 비율과 각 밸류 벡터를 곱해준 값을 모두 더하여 Self-Attention 층의 아웃풋을 내놓게 된다. 아래는 이 과정을 그림으로 나타낸 것이다.

![transformer_11](http://jalammar.github.io/images/t/self-attention-output.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

실제로는 벡터 하나마다 따로 계산이 진행되는 아니라 행렬을 사용하여 한꺼번에 계산하게 된다. 행렬이 계산되는 과정을 그림으로 보도록 하자. 아래는 인풋 토큰 벡터를 모두 포함하고 있는 행렬 $X$ 로부터 각각 쿼리 - 키 - 밸류에 해당하는 행렬 $Q, K, V$ 를 만들어내는 과정이다.

<p align="center"><img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation.png" alt="transformer_12" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

아래는 생성된 $Q, K, V$ 로부터 Self-Attention의 아웃풋 행렬인 $Z$ 를 만들어 내기 위한 연산 과정이다.

<p align="center"><img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" alt="transformer_13" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>

전체 과정을 다른 그림으로 나타내면 아래와 같이 나타낼 수 있다. 어떻게 Self-Attention이 진행되는지 쭉 따라가 보도록 하자.

<p align="center"><img src="http://jalammar.github.io/images/xlnet/self-attention-1.png" alt="transformer_15" style="zoom:50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/xlnet/self-attention-2.png" alt="transformer_15" style="zoom:50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/xlnet/self-attention-3.png" alt="transformer_15" style="zoom:50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/xlnet/self-attention-summary.png" alt="transformer_15" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Jay alammar Github</a></p>



























