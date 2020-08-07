---
layout: post
title: 어휘 분석과 구조 분석 (Lexical & Syntax Analysis) 
category: NLP
tag: NLP
---



본 포스트의 내용은 [고려대학교 강필성 교수님의 강의](https://www.youtube.com/watch?v=pXCHYq6PXto&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm) 와 [김기현의 자연어처리 딥러닝 캠프](http://www.yes24.com/Product/Goods/74802622) , [밑바닥에서 시작하는 딥러닝 2](http://www.yes24.com/Product/Goods/72173703) , [한국어 임베딩](http://m.yes24.com/goods/detail/78569687) 책을 참고하였습니다.



# 어휘 분석 (Lexical Analysis)

자연어처리(Natural Language Processing)의 대부분의 작업들은 기본 단위로 형태소(Morpheme)를 가집니다. 여기에서 말하는 형태소의 사전적 정의는 "의미를 가지는 요소로서는 더 이상 분석할 수 없는 가장 작은 말의 단위"입니다. 전체 말뭉치(Corpus)로부터 기본 단위인 형태소를 얻어내기 위해서는 어휘 분석(Lexical Analysis)이 필요합니다.

어휘 분석의 주요 태스크는 문장 분절, 토큰화(Tokenization), 품사 태깅(Part-of-Speech Tagging, POS Tagging)이 있습니다. 추가적으로 개체명 인식(Named Entity Recognition, NER) 이나 명사구 인식 등의 작업이 수행됩니다.



## 문장 분절 (Sentence splitting)

가장 먼저 **문장 분절(Sentence splitting)** 에 대해서 알아봅시다. 토픽 모델링(Topic Modeling)과 같은 몇 가지 태스크를 제외하면 문장 분절은 우리가 가장 먼저 수행해야 할 분석입니다. 문장을 분절할 때, 보통은 *"." / "?" / "!"* 등과 같은 문장 부호를 기준으로 나눕니다. 정제된 데이터라면 이런 문장 부호를 끝으로 문장이 끝나기 때문입니다. 하지만 이 문장 부호들이 항상 올바른 문장의 경계가 되는 것은 아닙니다.

다음의 예시들을 살펴봅시다. *"Mr.Lee is going home.", "원주율은 3.141592... 입니다."* 이 두 문장을 문장 부호를 기준으로 나누어 봅시다. 첫 번째 문장은 다음과 같이 분리될 것입니다. *["Mr.", "Lee is going home."]*    두 번째 문장은 다음과 같이 분리될 것입니다. *["원주율은 3.", "141592.", ".", ".", "입니다."]*    두 경우 모두 제대로 문장 분절이 이루어지지 않는 것을 볼 수 있습니다. 또는 인용구를 포함하는 문장도 문장 부호를 분절의 기준으로 삼는데 문제가 됩니다.

이런 문제 때문에 룰베이스 알고리즘이나 분리된 문장을 모델에 학습시키는 방법을 사용합니다. 자연어처리 패키지인 NLTK(Natural Language Toolkit) 등에서는 문장 분절 기능을 제공하고 있습니다. 아래는 *"자연어처리는 인공지능의 한 줄기 입니다. 시퀀스 투 시퀀스의 등장 이후로 딥러닝을 활용한 자연어처리는 새로운 전기를 맞이하게 되었습니다. 문장을 받아 단순히 수치로 나타내던 시절을 넘어, 원하는 대로 문장을 만들어낼 수 있게 된 것입니다."* 라고 적힌 파일을 문장별로 분절하는 예시 코드입니다. `nltk` 내에 있는 `sent_tokenize` 함수를 사용하여 문장을 분절하였습니다.

```python
import sys, fileinput, re
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
    for line in fileinput.input(files="gdrive/My Drive/nlp_exercise/sent_tokenize_ex.txt"):
        if line.strip() != "":
            line = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', line.strip())

            sentences = sent_tokenize(line.strip())

            for s in sentences:
                if s != "":
                    sys.stdout.write(s + "\n")
```

해당 코드를 실행하여 출력되는 값은 다음과 같습니다. 아래의 예시에서 원하는 대로 문장 분절이 된 것을 볼 수 있습니다.

*"자연어처리는 인공지능의 한 줄기 입니다.*
*시퀀스 투 시퀀스의 등장 이후로 딥러닝을 활용한 자연어처리는 새로운 전기를 맞이하게 되었습니다.*
*문장을 받아 단순히 수치로 나타내던 시절을 넘어, 원하는 대로 문장을 만들어낼 수 있게 된 것입니다."*



## 토큰화 (Tokenization)

문장 분절을 했다면 **토큰화(Tokenization)** 를 통해 더 자세한 어휘 분석을 할 수 있습니다. 토큰화란 텍스트를 이후 자연어처리에 사용되는 **토큰(Token)** 이라는 기본적인 단위로 나누는 과정입니다. 일반적으로 공개되어 있는 토큰 분석기(Tokenizer)를 사용합니다. 여러 개의 토큰 분석기가 있고 같은 문장을 분석하더라도 각 분석기마다 내놓는 토큰의 결과가 다릅니다. 그렇기 때문에 자신이 수행하려는 태스크에 맞는 분석기를 선택하여 사용하는 것이 중요합니다. 아래는 MC tokenizer와 Scan tokenizer로 특정 문서를 토큰화한 결과입니다.

두 분석기로 추출한 토큰의 결과가 다릅니다. 예를 들어, MC tokenizer로부터 추출된 토큰에는 공백이 제거되지 않고 숫자와 구두점, 특수문자 등은 모두 제거된 것을 볼 수 있습니다. 반대로 Scan tokenizer로부터 추출된 토큰에는 공백이 모두 제거되었으나 숫자와 구두점, 특수문자가 그대로 존재하는 것을 볼 수 있습니다.



![tokenizer](https://user-images.githubusercontent.com/45377884/81628131-cc033e00-943a-11ea-8582-303e3d175b92.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

하지만 토큰화도 문장 분절과 마찬가지로 어려움이 많습니다. 대표적인 예시로 *"pre-trained, C++", "A/C"* 과 같이 한 단어 내에 특수 문자가 있는 경우가 있습니다. 또한 *":-)", "ㅋㅋㅋㅋㅋㅋㅋㅋ"* 과 같이 이모티콘이나 구어체의 경우 토큰을 구별하기 어렵습니다. 중국어와 같은 특정 언어의 경우 공백이 존재하지 않아 토큰을 구분하기 어렵기도 합니다.

## 형태 분석 (Morphological analysis)

어휘 분석의 또 다른 작업으로 **형태 분석(Morphological analysis)** 이 있습니다. 형태 분석의 목적은 단어의 의미를 훼손하지 않는 선에서 단어를 정규화(Normalize)하는 것입니다. 정규화를 진행하면 말뭉치에 전체에 있는 텍스트 데이터의 차원을 줄일 수 있습니다. 예를 들어, *"I love her and she loves me."* 와 같은 문장을 토큰화 할 때 정규화를 하지 않으면, *"love"* 와 *"loves"* 는 다른 토큰으로 취급됩니다. 한 문장에서는 별 문제가 없어 보이지만 말뭉치 전체에 대해서 정규화를 해주지 않는다면 데이터의 차원이 매우 커지게 되고 Sparsity가 크게 늘어나는 문제가 발생합니다.

형태 분석에는 두 가지 방법, **어간 추출(Stemming)** 과 **표제어 추출(Lemmatization)** 이 있습니다. 먼저 어간 추출은 단어에서 정보를 이끌어 내는 데 중점을 두고 있는 분석 방법입니다. 어간 추출은 규칙 기반(Rule-based) 알고리즘을 사용하여 단어들의 기본 형태를 찾고 나머지 접사 부분을 잘라내는 데 기반을 두고 있습니다. 단어의 품사에 상관없이 의미의 기본 형태만 남기고 잘라내기 때문에 간단하고 빠르다는 장점이 있습니다. 하지만 규칙 기반 알고리즘의 특성상 분석하는 언어에 종속적이며 아예 다른 단어가 하나의 기본형으로 귀결되는 경우도 있어 의미(Semantic)를 보존하지 못하는 문제가 발생하기도 합니다.

표제어 추출은 품사를 보정해주며 정규화를 진행합니다. 아래 그림을 보면 어간 추출과 표제어 추출을 직접적으로 비교할 수 있습니다. 어간 추출은 단어의 품사가 어떻든지 동일하게 잘라내버립니다. 어간 추출에 의해서는 *"Innovation, Innovate, Innovative"* 가 모두 하나의 형태인 *"Innovat"* 로 잘려나가는 것을 볼 수 있습니다. 하지만 표제어 추출은 *"Innovation, Innovate, Innovative"* 의 품사를 모두 보존한 채로 정규화 하는 것을 볼 수 있습니다. 표제어 추출은 품사를 고려하는 작업 때문에 어간 추출보다 속도가 느리다는 단점이 있습니다. 하지만 단어의 의미를 잘 보존하며 이미 만들어진 사전을 기반으로 분석하기 때문에 오류가 적다는 장점도 가지고 있습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/81628366-767b6100-943b-11ea-9ae9-955f403e3b8a.png" alt="morpho" style="zoom: 50%;" /></p>



<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

두 방법 중 *"어떤 한 방법이 더 좋다"* 고 평가할 수 있는 지표는 없습니다. 필요한 태스크에 따라 어간 추출과 표제어 추출의 장단점을 잘 고려하여 사용하여야 합니다. 일반적으로 속도가 중요시되고 의미의 비중이 적은 정보 검색 등의 태스크에는 어간 추출이 많이 사용됩니다. 하지만 텍스트 마이닝에서 수행되는 대부분의 태스크에서는 단어의 의미가 중요시 되기 때문에 표제어 추출이 자주 사용되는 편입니다.



## POS Tagging

**품사 태깅(Part-of-Speech tagging, POS tagging)** 도 중요한 어휘 분석 작업 중 하나입니다. 품사 태깅은 POS Tagger를 사용하여 각 토큰에 해당하는 품사를 찾아주는 과정입니다. *"I love you.", "All you need is love."* 처럼 같은 단어( *"love"* ) 라도 다른 품사로 쓰일 수 있습니다. (앞 문장에서는 동사, 뒷 문장에서는 명사로 사용된 것을 볼 수 있습니다.)

머신러닝을 사용하여 품사 태깅을 하기 위해서는 수동으로 품사를 태깅한 말뭉치가 필요합니다. 이 때 훈련에 사용한 말뭉치와 비슷한 도메인 데이터에는 잘 작동하는 편이지만 다른 도메인에는 성능에 차이가 생기기도 합니다. 품사 태깅에는 의사 결정 트리(Decision Trees), 히든 마르코프 모델(Hidden Markov Models, HMM), 서포트 벡터 머신(Support Vector Machines, SVM) 등 다양한 머신러닝 알고리즘을 적용할 수 있습니다. 최근에는 Brill tagger 와 같은 트랜스포메이션 기반의 tagger가 많이 사용되고 있습니다.



## 개체명 인식 (Named Entity Recognition, NER)

품사 태깅과 마찬가지로 개체명 인식(Named Entity Recognition, NER)도 토큰 간의 관계를 파악하기 위한 어휘 분석 방법 중 하나입니다. 개체명 인식을 통해 텍스트 데이터 내에서 시간, 장소, 인물과 같은 특수한 개체(Entity)를 인식합니다. 예를 들어 *"Get me a flight from New York City to San Francisco for next Thursday."* 라는 문장이 있다고 해 봅시다. 해당 문장에 개체명 인식을 수행하면 *"New York City"* 라는 단어는 출발 장소로, *"San Francisco"* 라는 단어는 목적지로, *"Thursday"* 라는 단어는 출발 시각으로 미리 인식하여 구분할 수 있습니다.

개체명 인식은 크게 두 가지 방법으로 수행됩니다. 첫 번째는 각 개체에 대한 사전을 구축하는 방법이고 두 번째는 규칙 기반의 알고리즘을 사용하는 방법입니다. 사전 구축 방식에는 미리 리스트를 구축하여 같은 개체를 매칭하는 List Lookup 방식이 있습니다. 이는 단순하고 빠르다는 장점이 있지만 리스트를 관리하거나 업데이트 하기가 어렵다는 단점도 있습니다. 

개체명 인식을 위한 규칙 기반 알고리즘의 예시로는 Shallow parsing approach가 있습니다. 이 방법은 특정한 구조를 발견하면 그 구조에 따라 개체명을 할당하는 방식입니다. 예를 들어 *"Cap word + Street, Boulevard, Avenue, Crescent, Road"* 라는 구조를 발견하면 이것이 위치를 나타내는 개체임을 할당하게 됩니다.

이 두 가지 방법 외에도 머신러닝 기반의 알고리즘인 MITIE나 CRF++, CNN 등도 경우에 따라 사용하곤 합니다. 



# Syntax Analysis

**구조 분석(Syntax analysis)** 은 문장이 들어왔을 때 문법 형식에 맞도록 구조를 분석하는 과정입니다. 구조 분석에 사용되는 알고리즘은 파서(Parser)입니다. 파서 알고리즘은 두 가지 속성을 가지고 있습니다. 첫 번째 속성은 방향성(Directionality) 입니다. 이 방향성에 따라서 Top-down 방식으로 분석할 지, Bottom-up 방식으로 분석할 지를 결정하게 됩니다. 두 번째 속성은 탐색 전략(Search strategy)입니다. 이 속성은 트리를 옆으로 탐색할 것인지 아래쪽으로 파고들 것인지를 결정하게 됩니다. 이 두 특성에 따라 파서의 알고리즘이 달라지게 됩니다.

구조 분석을 표현하는 방식은 트리 방식과 리스트 방식이 있습니다. 아래 그림은 *"John ate the apple"* 을 분석할 경우 아래와 같이 나타낼 수 있습니다.

<img src="https://user-images.githubusercontent.com/45377884/81628477-b9d5cf80-943b-11ea-999e-316ff531f9ef.png" alt="syntax_rep" style="zoom: 50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

구조 분석을 한다고 해서 항상 한 가지 트리만 생성되는 것은 아닙니다. 언어의 모호함 때문에 어떤 알고리즘 방식을 채택하는지에 따라서 같은 문장에서도 다른 트리가 생성됩니다. 여기서 말하는 언어의 모호함에는 어휘로부터 발생하는 모호함(Lexical ambiguity)과 구조로부터 발생하는 모호함(Structural ambiguity)이 있습니다. 아래는 *"Time flies like an arrow"* 라는 문장에 구조 분석을 수행했을 때 어휘의 모호함 때문에 발생할 수 있는 두 가지 트리입니다.

<img src="https://user-images.githubusercontent.com/45377884/81628534-e12c9c80-943b-11ea-91d6-5a7a550d767d.png" alt="lex_ambi" style="zoom:50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>

구조로부터 발생하는 모호함에 의해서도 구조 분석 결과가 다르게 나타날 수 있습니다. 아래 이미지는 *"John saw Mary in the park"* 문장에 구조 분석을 수행한 결과를 트리로 나타낸 것입니다. 해당 문장만 보고는 *"John"* 이 *"Mary"* 를 공원 안에서 본 것인지 공원 밖에서 본 것인지 알 수 없습니다. *"Mary"* 의 위치에 따라 구조 분석한 결과가 달라지는 것을 볼 수 있습니다.

<img src="https://user-images.githubusercontent.com/45377884/81628583-00c3c500-943c-11ea-9a9d-f3f2190f9f64.png" alt="str_ambi" style="zoom:50%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://github.com/pilsung-kang/text-analytics">Text-Analytics Github</a></p>