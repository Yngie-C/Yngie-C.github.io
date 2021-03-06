---
layout: post
title: 파이토치로 배우는 자연어 처리
category: Book Review
tag: Book Review
---

 

# 자연어를, 파이토치로, 밑바닥부터

<img src="https://i.imgur.com/6Ncq23O.jpg" alt="NLP" style="zoom: 25%;" />



**"한빛미디어 <나는 리뷰어다> 활동을 위해서 책을 제공받아 작성된 서평입니다."**

## 이제는 토치를...!

몇 년 전까지만 해도 확실히 파이토치보다는 텐서플로우를 사용하는 곳이 많았습니다. 하지만 파이토치가 빠르게 성장한 덕분에 최근에는 비교할 수 있을만한 수준으로 올라왔습니다. 이러한 빠른 성장세의 원동력 중 하나는 자연어처리를 연구하는 쪽에서 파이토치를 지속적으로 많이 사용해 온 것이 아닐까 합니다. 자연어처리 논문을 구현한 코드나 모델을 활용하여 태스크를 수행한 많은 코드가 파이토치로 되어 있는 만큼, 이제 자연어처리 학습자에게 파이토치는 선택이 아니라 필수가 되어가는 느낌입니다. pytorch-lightning 이나 fast.ai 등이 나오면서 사용자에 대한 편의성도 점점 좋아지고 있지요.

이런 상황에서도 파이토치를 활용한 자연어처리 서적이 시중에 많지는 않았습니다. "김기현의 자연어처리 딥러닝 캠프" 정도가 파이토치를 사용한 자연어처리 서적 중 유명한 정도였습니다. 이번에 나온 "파이토치로 배우는 자연어처리"는 파이토치로 자연어처리를 시작하고자 하는 사람들에게 또 하나의 좋은 길잡이가 될 것으로 생각합니다. 기존에 있던 김기현님의 저서는 강의를 텍스트로 정리했기 때문에 파이토치를 조금 아는 사람이 보거나 강의와 같이 보아야 조금 원활하게 책장을 넘길 수 있었는데요. 이번에 리뷰한 서적은 코드가 조금 더 자세히 작성되어 있는 만큼 파이토치를 처음 접하는 사람이 책으로 공부하기에 조금 더 친절하게 느낄 수 있을 것으로 생각합니다. 실제로 함수가 구현된 부분마다 docstring과 주석이 상당히 자세하게 작성되어 함수의 역할과 이에 포함된 파라미터의 역할을 헷갈리지 않고 알 수 있다는 점이 좋았습니다.



## 역자의 노고가 느껴지는 부분들

이번 서적이 출간되면서 역자와 출판사의 노력이 들어간 부분도 많게 느껴졌습니다. 일단, 예제 코드가 모두 최신 버전의 파이토치에서 작성된 것이 좋았습니다. 원서가 나온지 꽤 시간이 흐른 만큼 서문을 보면 이전 버전(0.x)으로 작성되어 있다고 나와있는데요. 이번에 번역서를 출간하면서 예제 코드를 (출간 당시 최신 버전인) 1.8 에서 돌아가도록 작성하였습니다. 그럼에도 아직까지 예제 코드에서 별다른 오류가 발생하지 않은 것으로 보아 상당히 많은 점검이 있지 않았나 생각합니다. (모든 코드를 실행해 보지는 않았기 때문에 제가 실행해 보지 않은 다른 코드에서 오류가 발생할 수 있습니다.)

게다가 역서에만 있는 PORORO 라이브러리에 대한 설명도 인상깊습니다. 번역서인 자연어처리 서적은 모든 예제가 영어로 구성되어 있다는 점이 한편으로는 아쉬울 수 밖에 없는데요. 해당 서적의 부록에는 지난번 카카오브레인에서 발표한 한국어 자연어처리를 위한 라이브러리인 PORORO를 사용하여 태스크를 수행하는 방법을 소개하고 있습니다. 실제로 PORORO를 사용하면 상당히 많은 태스크를 쉽게 수행할 수 있는데요. 본 책에서는 이 중에서 OCR 부터 Image captioning, 기계 번역, 요약, 감성 분석, 추론, 토픽 분류를 소개하고 있습니다. 물론 부록에 할애한 분량이 많지 않기 때문에 자세하게 소개하고 있지는 않지만 번역서에서 이만큼이나 한국어 예제를 많이 수행해 볼 수 있다는 점이 인상깊었습니다.  



## 물론 아쉬운 점도 있습니다.

여러모로 아쉬운 점도 있습니다. 개인적으로는 "파이토치로 배우는 자연어처리(원서 제목 : Natural Language Processing with Pytorch)"와 "자연어처리로 배우는 파이토치"의 중간쯤에 위치하는 책이라고 생각합니다. 파이토치 코드는 자세하지만 이렇게 구현된 자연어처리 이론들에 대해서는 자세히 소개하고 있지 않다는 점이 아쉽기는 합니다. 그렇기 때문에 자연어처리를 처음 공부하고자 하는 사람에게 적합한 책은 아니라고도 생각합니다. 이런 부분에 대한 설명이 조금 부족한 만큼 다른 자연어처리 기본 서적을 겸하여 보거나, 기존에 자연어처리를 조금 공부한 사람들이 이를 파이토치로 실행해 보는 정도에서 적합한 책이 되지 않을까 하는 생각은 있습니다.

게다가 원서가 나온지 시간이 조금 지난만큼 최근 모델에 대한 부분이 많이 생략된 점도 아쉽습니다. 아무래도 최근에는 트랜스포머 이후 모델을 사용하여 자연어처리 태스크를 수행하는 비율이 늘어나고 있는데요. 번역서가 출간된 시점이 최근임에도 원서가 발행된 시점 때문에 이에 대한 부분이 많이 생략된 점이 아쉽다고 생각합니다. 하지만 해당 책으로 파이토치 사용법에 익숙해진다면 이후 부분에 대해서는 공개된 소스코드를 보면서도 충분히 공부할 수 있을 것으로 생각이 되기 때문에 (더구나 입문 서적에게는) 엄청난 문제가 아니라고 생각되는 부분이기도 합니다.