---
layout: post
title: 트랜스포머 (Transformer)
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# Transformer

2013년 [Word2Vec](https://yngie-c.github.io/nlp/2020/05/28/nlp_word2vec/)이 발표된 후 [GloVe, Fasttext](https://yngie-c.github.io/nlp/2020/06/03/nlp_glove/) 등이 등장하면서 단어 수준의 임베딩에 대한 방법론이 어느 정도 마무리 되었습니다. 그 뒤로는 문장 수준의 임베딩을 위한 방법론들이 등장하게 됩니다. 대표적인 것이 2017년 6월에 *["Attention is All You Need"(Ashish Vaswani et al.)](https://arxiv.org/abs/1706.03762)* 논문에서 발표한 **트랜스포머(Transformer)**입니다. 트랜스포머는 [Seq2Seq](https://yngie-c.github.io/nlp/2020/06/30/nlp_seq2seq/)에 등장했던 Attention 메커니즘을 극대화하기 위한 방법입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/the_transformer_3.png" alt="transformer_1"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

트랜스포머도 Seq2Seq모델과 유사한 인코더(Encoder)-디코더(Decoder) 구조를 가지고 있습니다. 아래는 트랜스포머의 전체적인 구조를 나타내는 이미지입니다. 이전까지의 Seq2Seq를 도식화한 이미지와 매우 유사하지만 Encoder**'s'**, Decoder**'s'**라는 점이 다르지요. 각각의 Encoders와 Decoders 내에는 복수의 Encoder, Decoder 블록이 들어있습니다.

<p align="center"><img src="http://jalammar.github.io/images/t/The_transformer_encoders_decoders.png" alt="transformer_2" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

## Structure

트랜스포머는 **인코더와 디코더 여러 개를 중첩**한 구조를 갖고 있다는 점에서 이전 모델과의 차이점을 갖습니다. 각각의 인코더와 디코더를 블록(Block)이라고 일컫습니다. 논문에서는 6개씩의 인코더 블록과 디코더 블록을 쌓은 구조를 사용하였습니다. 아래 이미지는 논문에서 발표한 트랜스포머의 구조를 도식화하여 나타낸 것입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png" alt="transformer_3" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

### Encoder

먼저 인코더의 내부 구조부터 살펴보겠습니다. 각 인코더 블록의 구조는 모두 동일합니다. 인코더 블록 내부에는 2개의 하위 레이어(Sub layer)가 있습니다. 데이터 흐름 순서대로 볼 때 첫 번째로 만나는 층이 **Self-Attention** 층이며, 이 층을 지나서 만나는 **FFNN(Feed forward neural network)** 층입니다. Self-Attention 층에서는 한 토큰이 다른 토큰과 얼마나 관련이 있는 지에 관한 가중치를 계산합니다.

<p align="center"><img src="http://jalammar.github.io/images/t/Transformer_encoder.png" alt="transformer_4" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

다음으로 디코더를 보겠습니다. 디코더는 총 3개의 하위 레이어를 가지고 있습니다. 데이터가 흐르는 순서대로 볼 때 첫 번째로 만나는 층은 **(masked) Self-Attention** 층입니다. 두 번째로 만나는 층은 **Encoder-Decoder Attention** 층입니다. 인코더 블록에는 없었던 부분입니다. 인코더-디코더 Attention 층에서는 인코더가 처리한 정보를 받아서 Attention 메커니즘을 수행합니다. 마지막은 인코더와 동일하게 **FFNN** 층으로 구성되어 있습니다. 아래 그림은 인코더 블록을 구성하고 있는 2개의 하위 레이어와 디코더 블록을 구성하고 있는 3개의 하위 레이어를 도식화하여 나타낸 것입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/Transformer_decoder.png" alt="transformer_5"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

## Input Embedding

트랜스포머의 전체적인 구조는 아래 이미지와 같습니다. 아래에서는 이미지를 단순화하기 위해 인코더와 디코더 블록 하나씩만을 나타내었으나, 실제로는 $N$(논문에서는 6)개의 블록 스택을 쌓아 구성합니다.

<p align="center"><img src="https://miro.medium.com/max/1000/1*o0pS0LbXVw7i41vSISrw1A.png" alt="transformer_6" style="zoom:120%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://mc.ai/transformer-self-attention-part-1/">mc.ai</a></p>

먼저 **입력 임베딩(Input embedding)**에 대하여 알아보겠습니다. 입력 임베딩은 입력 시퀀스의 아이템을 첫 번째 인코더 블록에 입력할 수 있는 벡터로 만드는 작업입니다. 일반적으로는 Word2Vec, GloVe, Fasttext 등에 의해 사전 학습된 임베딩 벡터를 사용합니다. 트랜스포머의 인코더와 디코더는 512차원의 벡터를 다룹니다. 인풋 임베딩에 들어가기 위해서는 512차원의 임베딩 벡터가 필요합니다. 얼마만큼의 아이템을 하나의 시퀀스로 처리할 것인지에 대해서는 하이퍼파라미터를 통해서 조정할 수 있습니다. 일반적으로는 가장 긴 문장의 길이를 사용하고 짧은 문장에 대해서는 패딩(`<pad>`)을 처리해줍니다.

## Positional Encoding

다음으로 필요한 작업은 **포지셔널 인코딩(Positional encoding)**입니다. Self-Attention 에서는 RNN처럼 순차적으로 토큰을 다루지 않습니다. 그렇기 때문에 단어의 순서 정보를 완전히 무시되지요. 하지만 포지셔널 인코딩을 사용하면 완벽히 복구할 수는 없더라도 단어의 상대적인 위치를 어느 정도 보완해줄 수 있습니다. 그렇다면 포지셔널 인코딩은 어떤 조건을 만족해야 할까요?

일단, 각 위치마다 유일한 값을 출력해야 합니다. 두 단어의 위치가 같을 수는 없겠지요. 그리고 길이가 다른 문장의 단어 위치를 나타낼 때에도 단어의 상대적인 거리가 같으면 같은 차이를 보여야 합니다. 아래의 예시를 보며 이야기 해보도록 하겠습니다.

> "어머님 나 는 별 하나 에 아름다운 말 한마디 씩 불러 봅니다"
>
> "소학교 때 책상 을 같이 했 던 아이 들 의 이름 과 패 경 옥 이런 이국 소녀 들 의 이름 과 벌써 애기 어머니 된 계집애 들 의 이름 과 가난 한 이웃 사람 들 의 이름 과 비둘기 강아지 토끼 노새 노루 프란시스 쟘 라이너 마리아 릴케 이런 시인 의 이름 을 불러 봅니다"

첫 번째 문장에서 *"나"*와 *"아름다운"*은 사이에 4개의 형태소를 두고 있습니다. 마찬가지로 두 번째 문장에서도 *"책상"*과 *"아이"*는 4개의 형태소를 사이에 두고 떨어져 있네요. 이런 경우에 상대적인 단어의 거리가 같다고 할 수 있습니다. 그렇다면 *"1, 2, 3, 4 ..."* 처럼 단어 순서대로 고유한 값을 부여하면 안될까요? 이런 방식을 사용하여 각 문장의 포지셔널 인코딩을 출력하면 아래와 같습니다.

> 1: '어머님', 2: '나', 3: '는', 4: '별', 5: '하나', 6: '에', 7: '아름다운', 8: '말', 9: '한마디', 10: '씩', 11: '불러', 12: '봅니다'
>
> 1: '소학교', 2: '때', 3: '책상', 4: '을', 5: '같이', 6: '했', 7: '던', 8: '아이', 9: '들', 10: '의', 11: '이름', 12: '과', 13: '패', 14: '경', 15: '옥', 16: '이런', 17: '이국', 18: '소녀', 19: '들', 20: '의', 21: '이름', 22: '과', 23: '벌써', 24: '애기', 25: '어머니', 26: '된', 27: '계집애', 28: '들', 29: '의', 30: '이름', 31: '과', 32: '가난', 33: '한', 34: '이웃', 35: '사람', 36: '들', 37: '의', 38: '이름', 39: '과', 40: '비둘기', 41: '강아지', 42: '토끼', 43: '노새', 44: '노루', 45: '프란시스', 46: '쟘', 47: '라이너', 48: '마리아', 49: '릴케', 50: '이런', 51: '시인', 52: '의', 53: '이름', 54: '을', 55: '불러', 56: '봅니다'

위와 같이 인코딩 값을 부여하게 되면 긴 문장에서 맨 뒤에 위치한 토큰의 값이 매우 커집니다. 포지셔널 인코딩 값이 매우 커지게 되면 원래의 인풋 임베딩 값에 영향을 주게 되지요. 그렇다면 범위를 정해놓고 등분하여 나타내면 되지 않을까요? $[0,1]$ 범위를 단어 개수만큼 등분하여 포지셔널 인코딩 값을 나타내어 보겠습니다.

> 0: '어머님', 0.091: '나', 0.182: '는', 0.273: '별', 0.364: '하나', 0.455: '에', 0.545: '아름다운', 0.636: '말', 0.727: '한마디', 0.818: '씩', 0.909: '불러', 1: '봅니다'
>
> 0: '소학교', 0.018: '때', 0.036: '책상', 0.055: '을', 0.073: '같이', 0.091: '했', 0.109: '던', 0.127: '아이', 0.145: '들', 0.164: '의', 0.182: '이름', 0.2: '과', 0.218: '패', 0.236: '경', 0.255: '옥', 0.273: '이런', 0.291: '이국', 0.309: '소녀', 0.327: '들', 0.345: '의', 0.364: '이름', 0.382: '과', 0.4: '벌써', 0.418: '애기', 0.436: '어머니', 0.455: '된', 0.473: '계집애', 0.491: '들', 0.509: '의', 0.527: '이름', 0.545: '과', 0.564: '가난', 0.582: '한', 0.6: '이웃', 0.618: '사람', 0.636: '들', 0.655: '의', 0.673: '이름', 0.691: '과', 0.709: '비둘기', 0.727: '강아지', 0.745: '토끼', 0.764: '노새', 0.782: '노루', 0.8: '프란시스', 0.818: '쟘', 0.836: '라이너', 0.855: '마리아', 0.873: '릴케', 0.891: '이런', 0.909: '시인', 0.927: '의', 0.945: '이름', 0.964: '을', 0.982: '불러', 1: '봅니다'

이렇게 되면 문장 길이, 즉 문장을 구성하는 단어 개수에 따라 상대적인 거리 관계가 깨지게 됩니다. 첫 번째 문장에서 *"어머님"*과 *"나"*, 두 번째 문장에서 *"소학교"*와 *"때"*는 모두 바로 옆 단어이지만 첫 번째 문장에서의 차이는 $0.091$이고 두 번째 문장에서는 $0.018$으로 5배나 차이나지요. 그리하여 논문에서는 이런 조건을 모두 만족하는 sinusoid를 사용했습니다. sinusoid 포지셔널 인코딩 함수의 수식은 다음과 같습니다.

$$
\begin{aligned}
PE_{(\text{pos}, 2i)} &= \sin(\text{pos}/10000^{2i/d_{\text{model}}}) \\
PE_{(\text{pos}, 2i+1)} &= \cos(\text{pos}/10000^{2i/d_{\text{model}}})
\end{aligned}
$$

아래의 그림은 위 함수를 사용하여 사용하여 포지셔널 인코딩한 단어 사이의 거리를 시각화한 것입니다. 왼쪽 그래프는 10개의 단어로 이루어진 문장의 모든 단어 사이의 거리를, 오른쪽 그래프는 50개의 단어로 이루어진 문장에서 앞부분 10개의 단어 사이의 거리를 시각화했습니다. 차원은 논문에서 사용한 $d_\text{model} = 512$ 를 사용하였습니다.

<img src="https://user-images.githubusercontent.com/45377884/91164716-824d1380-e70a-11ea-8279-f4c2b8d6e0f6.png" alt="pos_encode"  />

두 그래프로부터 문장의 길이가 달라지더라도 단어 사이의 거리는 보존됩니다. 문장의 단어가 몇 개이든 항상 바로 옆 단어는 3.714만큼, 그 옆 단어는 6.967만큼 차이가 나게 됩니다. 게다가 문장의 길이가 길어져도 포지셔널 인코딩 값이 한없이 커지지 않습니다. 멀리 떨어질수록 증가폭이 점점 떨어지기 때문에 *"W1"*과 *"W2"*사이의 거리는 3이상이지만, *"W1"*과 *"W10"*사이의 거리는 12.37로 9배의 차이가 나지 않는 것을 볼 수 있습니다. 증가폭이 점점 줄기 때문에 *"W1"*과 *"W20"*사이의 거리를 구해보아도 13.98밖에 되지 않습니다.

이렇게 구한 포지셔널 인코딩 벡터는 단어의 임베딩 벡터에 더해진 후에 인코더 블록으로 들어가게 됩니다.

## Encoder Block

인코더 블록의 각 하위 레이어에서 어떤 일이 일어나는 지에 대해서 알아보겠습니다. 아래는 첫 번째 인코더 블록에 들어가는 벡터 $x_1, x_2$가 어떤 과정을 거치는 지를 나타낸 이미지입니다. 

<p align="center"><img src="http://jalammar.github.io/images/t/encoder_with_tensors_2.png" style="zoom:66%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

위에서 말했듯이 인코더 블록에서 **Self-Attention** 하위 레이어를 먼저 만나게 됩니다. 이를 거쳐 나온 출력 벡터 $z_1, z_2$는 각각 **FFNN**층을 거쳐 인코더 블록을 빠져나오게 됩니다. 위 그림에서도 볼 수 있는 것처럼 Self-Attention층에서는 각각의 입력 벡터가 서로 영향을 끼치는(Dependency) 관계에 있습니다. 하지만 FFNN에서는 따로 들어가기 때문에 서로 영향을 끼치지 않지요. 그렇기 때문에 FFNN에서는 병렬화가 가능하고 덕분에 더 빠르게 학습할 수 있습니다. 인코더 블록가 출력하는 벡터 $r_1, r_2$는 그 다음 인코더 블록의 인풋으로 다시 사용됩니다.

### Self-Attention

다음의 예시 문장을 통해서 Self-Attention 레이어에서 어떤 일이 일어나는지 알아보겠습니다.

$$
\text{"The animal didn't cross the street because it was too tired"}
$$

사람은 위 문장에서의 *"it"* 이 어떤 단어를 지칭하는지 쉽게 알 수 있습니다. 하지만 컴퓨터가 이를 맞추기는 쉽지 않습니다. 이런 지시대명사 외에도 문장에 있는 단어는 시퀀스 내에 있는 다른 단어와 관계를 맺고 있습니다. 이 관계를 표현하는 것이 바로 Self-Attention 레이어의 목적입니다. 아래는 5번째 인코더 블록에서 *"it"* 이 어떤 단어와 가장 깊은 관계를 맺고 있는 지를 나타내는 그림입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/transformer_self-attention_visualization.png" alt="transformer_8" style="zoom:110%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

위 그림에서 *"it"*이 *"The animal"*을 정확히 가리키고 있는 것을 볼 수 있습니다. Self-Attention층은 어떤 과정을 통해 이를 알아낼 수 있는 것일까요? Self-Attention층 내부에서 일어나는 일에 대해서 알아보겠습니다.

먼저 인풋 벡터가 들어오면 Self-Attention층에서는 세 가지 벡터를 인풋 벡터의 수만큼 준비합니다. 이 세 가지 벡터는 각각 **쿼리(Query, Q), 키(Key, k), 밸류(Value, V)**라고 부릅니다. 가장 먼저 쿼리는 현재 처리하고자 하는 토큰을 나타내는 벡터입니다. 키는 일종의 레이블(label)로, 시퀀스 내에 있는 모든 토큰에 대한 아이덴티티(Identity)를 나타냅니다. 마지막으로 밸류는 키와 연결된 실제 토큰을 나타내는 벡터입니다. 아래는 쿼리, 키, 밸류의 특징을 잘 살린 그림입니다.

<p align="center"><img src="http://jalammar.github.io/images/gpt2/self-attention-example-folders-3.png" alt="transformer_9" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

위 그림에서 `Query #9` 벡터는 10번째 단어인 *"it"*에 연관된 벡터입니다. 컴퓨터는 모든 단어의 키를 살펴보며 쿼리와 가장 잘 맞는 것을 찾습니다. 그리고 해당하는 키에 걸려있는 실제 밸류를 가져가게 되지요. 그렇다면 각각의 쿼리, 키, 밸류 벡터는 어떻게 만들어지는 것일까요?

Self-Attention 층 내부에는 세 벡터를 만들기 위한 행렬 $W^Q, W^K, W^V$ 가 초기화된 값으로 미리 준비되어 있습니다. 입력되는 토큰 벡터는 각 행렬과의 곱을 통해서 그에 맞는 쿼리, 키, 밸류 벡터를 만들게 됩니다. 아래 그림은 미리 준비된 행렬로부터 각 벡터가 만들어짐을 보여주는 이미지입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/transformer_self_attention_vectors.png" alt="transformer_10" style="zoom: 80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

위 그림에서 $\mathbf{x}_1, \mathbf{x}_2$는 토큰 벡터입니다. 수식으로 나타내면 아래와 같이 되지요.


$$
\begin{aligned}
\mathbf{x}_1 \cdot W^Q = \mathbf{q}_1 \qquad& \mathbf{x}_1 \cdot W^K = \mathbf{k}_1 & \mathbf{x}_1 \cdot W^V = \mathbf{v}_1 \\
\mathbf{x}_2 \cdot W^Q = \mathbf{q}_2 \qquad& \mathbf{x}_2 \cdot W^K = \mathbf{k}_2 & \mathbf{x}_2 \cdot W^V = \mathbf{v}_2
\end{aligned}
$$


$W$행렬에 해당하는 $W^Q, W^K, W^V$의 각 요소 값은 학습 과정에서 갱신되는 파라미터입니다. 이후에 등장할 Multi-head Attention을 사용한다면 쿼리, 키, 밸류 벡터의 차원은 인풋 토큰 벡터보다 작아집니다. 논문에서는 8개의 Head를 사용하기 때문에 512차원의 인풋 벡터를 사용하고 쿼리, 키, 밸류 벡터는 64차원으로 설정하였습니다. 자세한 내용은 아래 Multi-head Attention에서 알아보겠습니다.

쿼리와 잘 맞는 키를 찾는 과정은 말 그대로 **어텐션(Attention)**입니다. 벡터의 곱연산(Dot product)을 기반으로 Softmax해주어 연관 비중을 구하게 되지요.

<p align="center"><img src="http://jalammar.github.io/images/gpt2/self-attention-example-folders-scores-3.png" alt="transformer_10" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

논문에서는 학습시 그래디언트를 안정적으로 만들어주기 위해서 내적 값을 벡터 차원 수의 제곱근인 $\sqrt{d_k}$로 나누어주었습니다. 최종적으로는 소프트맥스를 통해 구해준 비중과 밸류 벡터를 곱해준 값을 모두 더하여 Self-Attention 층의 최종 출력 벡터로 계산하게 됩니다. 아래는 Self-Attention 층의 과정을 그림으로 나타낸 것입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/self-attention-output.png" alt="transformer_11"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>



위 그림에서 *"Thinking"*은 자기 자신과 88%만큼의 관계를, *"Machine"*과는 12%만큼의 관계를 맺고 있는 것을 볼 수 있습니다. 각 값을 밸류 벡터인 $\mathbf{v}_1, \mathbf{v}_2$와 각각 곱하여 더하면 최종 출력 벡터인 $\mathbf{z}_1$이 나오게 됩니다.

실제 계산은 행렬을 사용하여 한꺼번에 수행합니다. 아래는 토큰 벡터로 이루어진 행렬 $X$ 로부터 쿼리, 키, 밸류에 해당하는 행렬 $Q, K, V$ 를 만드는 과정입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation.png" alt="transformer_12" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

아래는 생성된 $Q, K, V$ 로부터 Self-Attention의 아웃풋 행렬인 $Z$ 를 만들어 내기 위한 연산 과정입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" alt="transformer_13" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

전체 과정을 다른 그림으로 나타내면 아래와 같이 나타낼 수 있습니다. Self-Attention층 내에서 어떤 과정이 진행되는지를 그림을 따라가면서 알아보도록 하겠습니다.

<p align="center"><img src="http://jalammar.github.io/images/xlnet/self-attention-1.png" alt="transformer_15" style="zoom:50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/xlnet/self-attention-2.png" alt="transformer_15" style="zoom:50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/xlnet/self-attention-3.png" alt="transformer_15" style="zoom:50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/xlnet/self-attention-summary.png" alt="transformer_15" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

### Multi-Head Attention

다음으로는 **멀티헤드 어텐션(Multi-Head Attention)**를 알아보겠습니다. Multi-Head Attention은 단어 간의 관계를 여러 번 계산하기 위해서 사용합니다. 동전을 던져 앞면이 나오는 확률을 구한다면 2번 던질 때보다 20번 던질 때 평균에 가까워지겠지요. 논문에서 사용한 Head의 개수는 $8(=512/64)$ 입니다. 총 8번의 Self-Attention을 실행하여 8개의 아웃풋 $Z_0, Z_1, \cdots , Z_7 $ 을 만들어냅니다. 아래는 이 과정을 그림으로 나타낸 것입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/transformer_attention_heads_z.png" alt="transformer_16" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

이렇게 나온 각각의 아웃풋 행렬 $Z_n (n=1,\cdots,7)$은 이어붙여(Concatenate)집니다. 또 다른 파라미터 행렬인 $W^o$ 와의 내적을 통해 Multi-Head Attention의 최종 결과인 행렬 $Z$를 만들어냅니다. 여기서 행렬 $W^o$의 요소 역시 학습을 통해 갱신됩니다. 최종적으로 생성된 행렬 $Z$는 토큰 벡터로 이루어진 행렬 $X$와 동일한 크기(Shape)가 됩니다. 이렇게 행렬 $Z$는 이제 FFNN으로 넘어가게 됩니다. 아래는 이 과정을 그림으로 나타낸 것입니다.

<p align="center"><img src="http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png" alt="transformer_17" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

## Residual Block & Layer Normalization

Self-Attention층에서 출력된 행렬(벡터)은 FFNN층으로 가기 전에 Residual block과 Layer normalization 과정을 거치게 됩니다. Residual Block이란 Self-Attention층을 통과한 출력 벡터에 원래의 입력 벡터를 더해주는 과정입니다. 이렇게 더해주면 역전파(Backpropagation)에서 그래디언트를 항상 1이상으로 유지하기 때문에 정보를 더 잘 보존합니다. Computer vision분야의 ResNet에서도 Residual block을 사용하여 성능을 끌어올린 사례가 있습니다.

<img src="https://miro.medium.com/max/570/1*D0F3UitQ2l5Q0Ak-tjEdJg.png" alt="res_block" style="zoom:80%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec">towardsdatascience.com</a></p>

Residual Block을 지나면 Layer Normalization을 거쳐 드디어 FFNN에 들어갑니다. 아래 그림은 Residual block(Add)과 Layer normalization 과정이 인코더 블록의 Self-Attention과 FFNN 사이에서 수행됨을 잘 보여주고 있습니다. 

<p align="center"><img src="http://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png" alt="transformer_18" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

## FFNN(Feed forward neural network)

다음으로 벡터는 FFNN(Feed forward neural network)로 들어갑니다. 동일한 인코더 블록 내에 있는 FFNN은 서로 동일한 가중치를 공유합니다. FFNN에서 일어나는 계산은 아래와 같은 수식에 근거하여 계산됩니다.



$$
FFNN(x) = \max(0, xW_1 + b_1) W_2 +b_2
$$



아래 그림은 FFNN 입력층(Input layer), 은닉층(Hidden layer), 출력층(Output layer)에서의 벡터를 그림으로 나타낸 것입니다. 512차원의 입력 벡터가 활성화 함수(Activation function)인 ReLU, 즉 $\max(0,\text{input})$을 지나 2048차원의 벡터가 된 뒤에 다시 512차원의 벡터로 출력됩니다.

![ffnn_1](https://user-images.githubusercontent.com/45377884/86258473-d4cfef00-bbf5-11ea-86ce-4019c22178a3.png) 

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text analytics 강의자료</a></p>

## Masked Self Attention

인코더 블록의 내부 구조를 모두 알아보았으니 이제는 디코더 블록의 내부 구조를 살펴볼 차례입니다. 디코더의 첫 번째 하위 레이어(Sub layer)는 인코더와 동일한 Self-Attention입니다. 하지만 디코더의 Self-Attention에는 특별한 조건이 하나 더 붙습니다. 디코더는 시퀀스를 출력하는 역할을 하기 때문에 등장하지 않은 단어에 대한 계산을 해서는 안됩니다.

예를 들어 *"나는 학생이다.(I am a student)"*라는 문장을 번역하는데 디코더가 *"I am"* 뒤에 올 단어를 예측할 차례라고 합시다. 3번째 단어를 예측하는 데 뒤에 등장하는*"student"*를 보고 결정해서는 안된다는 이야기입니다. 이 때문에 디코더에서 Self-Attention을 계산할 때에는 예측하련느 토큰 이전에 있는 토큰에 대한 정보만 남기고 마스킹(Masking)을 해줍니다. 이런 이유 때문에 디코더의 Self-Attention은 특별히 **Masked Self-Attention**이라고 부릅니다.

*"나는 학생이다.(I am a student)"*라는 문장을 번역하는데 *"I am"* 뒤에 올 단어를 예측할 차례라면 *"a", "student"*에 대한 정보는 얻을 수 없도록 가려버리는 것이지요. 아래는 인코더에서 수행하는 Self-Attention과 디코더에서 수행하는 Masked Self-Attention의 차이를 그림으로 나타낸 것입니다.

<p align="center"><img src="http://jalammar.github.io/images/gpt2/self-attention-and-masked-self-attention.png" alt="transformer_19" style="zoom: 50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

그렇다면 어떻게 정보를 마스킹 해줄 수 있을까요? 아래는 위 그림에 있는 두 과정의 계산 차이를 나타낸 그림입니다. 아래 그림을 보며 설명을 이어가겠습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/86260921-e4046c00-bbf8-11ea-8c7c-f8b58b8b46e9.png" alt="masked_attn"/></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text analytics 강의자료</a></p>

위 그림을 보면 디코더에서 *"Thinking"*이라는 단어를 예측할 때에는 그 뒤에 있는 *"Machine"*이라는 단어의 쿼리-키 내적값 $q \cdot k$의 값을 $-\infty$로 만들어주는 것을 볼 수 있습니다. 이렇게 되면 마스킹된 부분은 소프트맥스 함수에 넣어주어도 그 값이 0이 되고, 이 때문에 밸류 벡터가 아무런 영향을 미치지 못하게 됩니다. 아래는 4개 입력 벡터에 대해 Masked Self-Attention이 작동하는 과정을 도식화한 그림입니다.

<p align="center"><img src="http://jalammar.github.io/images/xlnet/masked-self-attention-2.png" alt="transformer_20" style="zoom: 50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

"robot must obey orders"라는 4단어로 이루어진 문장에 대해서 Masked Self-Attention을 수행하는 되는 전체 과정은 아래와 같이 진행됩니다. 

<p align="center"><img src="http://jalammar.github.io/images/gpt2/transformer-decoder-attention-mask-dataset.png" alt="transformer_21" style="zoom: 50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/gpt2/queries-keys-attention-mask.png" alt="transformer_21" style="zoom:50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/gpt2/transformer-attention-mask.png" alt="transformer_22" style="zoom:50%;" /></p>

<p align="center"><img src="http://jalammar.github.io/images/gpt2/transformer-attention-masked-scores-softmax.png" alt="transformer_23" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

## Encoder-Decoder Attention

Masked Self-Attention층의 출력 벡터는 인코더 블록에서와 동일하게 Residual block과 Layer normalization을 거친 뒤에 **인코더-디코더 Attention(Encoder-Decoder Attention)**과정을 거치게 됩니다. 이 층에서는 인코더의 마지막 블록에서 출력된 키, 밸류 행렬으로 Self-Attention 메커니즘을 한 번 더 수행합니다.

<p align="center"><img src="http://jalammar.github.io/images/t/transformer_decoding_2.gif" alt="transformer_24"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>

## Linear & Softmax Layer

모든 인코더와 디코더 블록을 거친 벡터는 최상단 층인 Linear 층과 소프트맥스 층을 차례대로 거칩니다. Linear 층은 단순한 완전 연결(Fully connected) 신경망이며 가장 마지막 디코더 블록의 출력 벡터를 로짓 벡터(Logit vector)로 변환해 줍니다. 그리고 소프트맥스 층에서는 이 로짓 벡터를 각 토큰이 위치할 확률로 바꾸어 줍니다. 이 과정을 그림으로 나타내면 아래와 같습니다.

<p align="center"><img src="http://jalammar.github.io/images/t/transformer_decoder_output_softmax.png" alt="transformer_25" style="zoom: 80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="http://jalammar.github.io/illustrated-transformer/">jalammar.github.io</a></p>



## Complexity & Performance

### Complexity

트랜스포머의 구조는 위에서 모두 살펴보았습니다. 그렇다면 이러한 트랜스포머 모델의 **복잡도(Complexity)**는 어떠한지 다른 신경망 구조와 비교하여 보도록 하겠습니다.

| Layer Type                  | Complexity per Layer   | Sequential Operations | Maximum Path Length |
| --------------------------- | ---------------------- | --------------------- | ------------------- |
| Self Attention              | $O(n^2 \cdot d)$       | $O(1)$                | $O(1)$              |
| Recurrent                   | $O(n \cdot d^2)$       | $O(n)$                | $O(n)$              |
| Convolutional               | $O(k \cdot n \cdot d)$ | $O(1)$                | $O(\log_k (n))$     |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$                | $O(n / r)$          |

Self-Attention 메커니즘은 Sequential하게 입력을 받지 않으므로 이 부분에서 RNN보다 시간복잡도 측면에서 유리한 것을 볼 수 있습니다.

### Performance

BLEU Score[^1]를 통해서 다른 모델과 비교한 트랜스포머의 성능은 다음과 같습니다. 트랜스포머의 크기를 키운 Big 모델은 BLEU 기준으로 당시 SOTA(State-of-the-art)를 기록하였습니다. 트랜스포머 모델이 이전의 방법을 사용하였을 때보다 더 적은 학습 비용을 필요로 하는 것도 볼 수 있습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/86265841-4c564c00-bbff-11ea-995a-c8049836a356.png" alt="trans_perform_1" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://arxiv.org/abs/1706.03762">Attention is all you need</a></p>

게다가 논문에서는 입력 임베딩 벡터의 차원 $d_\text{model}$이나 Self-Attention 내부 쿼리-키-밸류 벡터의 차원인  $d_k$, 인코더(디코더) 블록의 개수인 $N$ 등을 변형시켜 가면서 측정한 성능을 PPL(Perplexity)와 BLEU 기준으로 보여주고 있습니다. 아래는 위와 같은 실험 내용에 대한 이미지입니다. 

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/86266384-1c5b7880-bc00-11ea-8c56-48405bbb625a.png" alt="trans_variation" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://arxiv.org/abs/1706.03762">Attention is all you need</a></p>

[^1]: [김동화님 블로그](https://donghwa-kim.github.io/BLEU.html) 에 잘 정리되어 있습니다.