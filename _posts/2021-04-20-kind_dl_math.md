---
layout: post
title: 친절한 딥러닝 수학
category: Book Review
tag: Book Review
---

 

# 친절한(?) 딥러닝 수학

**"한빛미디어 <나는 리뷰어다> 활동을 위해서 책을 제공받아 작성된 서평입니다."**

<p align="center"><img src="https://image.aladin.co.kr/Community/paper/2021/0420/pimg_7235241502919394.jpg" alt="serious_python" style="zoom: 33%;" /></p>



'수학'. 이름만으로도 여럿 까다롭게 만드는 존재입니다. 4차 산업혁명이라는 키워드에 힘입어 많은 사람들이 딥러닝 공부에 도전하고 있습니다. 하지만 딥러닝을 공부하다보면 '수학'이라는 녀석을 피할 수 없게 됩니다. 가중치와 편향이 곱해지면서 더해지는 순전파, 그리고 이들이 어떻게 학습되는지에 대한 역전파를 이해하려면 (특정 학과를 제외하고는) 고등학교 이후 듣지 않고 지내던 '미분'이라는 단어를 만나기 마련이지요. 가중치마다 어떻게 학습되는지를 알기 위해서 공부해야 하는 '편미분'과 연쇄 법칙(Chain rule)은 보너스(?)입니다. 딥러닝 입문자 입장에서는 코딩 따라가기도 바쁜데 갑작스레 등장하는 미분에 난색을 표할 수 밖에 없습니다. 물론 몰라도 코딩하는데 큰 문제가 없을 수도 있고 수학 수식을 건너뛰며 딥러닝 공부를 할 수도 있을 것입니다. 하지만 이후에 어떤 구조가 등장하고 이러한 구조가 왜 등장하였는지를 이해하기 위해서 다시 돌아와 만나야 하는 것 역시 '수학'입니다.



## 친절한 딥러닝 수학

물론 수학을 제대로 공부해나가고자 한다면 전공 서적을 보는 것만큼 좋은 게 없겠지만, '미분' 한 단어도 두려운 입문자에게 전공 서적이란 인테리어 소품이 되기 십상입니다. (방 안 책꽂이에 박혀 나올 생각이 없는 전공 책을 바라보며...)

대화체로 구성된 이 책은 퍼셉트론의 작동 원리부터 자세히 수식을 설명하며 전개됩니다. 가장 도움을 많이 받을 수 있는 부분은 순전파와 역전파에 대한 설명인데요. 신경망이 순전파를 통해 어떻게 예측값을 내놓고, 계산된 예측값으로부터 어떤 방식으로 가중치가 갱신되는 지를 자세하게 설명하고 있어서 좋았습니다. 대화체로 쓰여 있어서 가독성도 좋았고 역전파에서 각 층마다 신경망 이미지를 곁들여 수식을 설명해주는 방식이 마음에 들었습니다. 여기까지는 순전파와 역전파에 대해 수학적 개념이 흐릿한 분들에게 정말 좋은 책이 될 것 같다는 생각이 들었습니다.



## 친절한(?) 딥러닝 수학

문제는 그 다음부터입니다. 역전파 다음 챕터인 Chap 4. 에서는 합성곱 신경망(CNN)에 대해 수학 수식을 통해서 다루고 있습니다. CNN의 계산이 워낙 많다보니 식이 꽤나 복잡해지는데요. 물론 책에서의 설명처럼 "항이 많을 뿐 하나씩 따라가면 똑같"기는 합니다. 하지만 처음 배우는 사람에게는 이런 수식이 되려 난감하게 느껴지지 않을까 하는 생각이 들었습니다. 시그마 갯수도 많아지고, 각 커널을 설명하기 위한 첨자도 많아지니 수학 기호에 익숙지 못한 이들에게는 이러나 저러나 이해하기 힘든 식이 되어버리는 셈이지요. 자세한 것은 좋지만 과한 상세함이 주는 일종의 불편함이 느껴졌습니다. 게다가 식이 길어지다보니 합성곱 신경망을 설명하기 위해서 할애한 지면이 거의 책의 절반정도가 된다는 점도 아쉽습니다. 게다가 선형대수, 즉 벡터 및 행렬 계산에 대한 설명은 조금 부족하지 않나 하는 생각도 들었습니다.

책의 마지막은 신경망 날코딩 구현으로 끝나는데 이 부분에서 다른 책과 차별점이 별로 없어진다는 생각이 들었습니다. 오히려 머신러닝, 딥러닝 공부에 필요한 '수학'을 공부하고자 하시는 분이라면 동일 출판사의 '김도형의 데이터 사이언스 스쿨'이나 길벗의 '프로그래밍을 위한 ~' 시리즈를 추천드립니다. 그리고 날코딩으로 신경망을 구현하고자 하는 분들에겐 역시 '밑바닥부터 시작하는 딥러닝'만한 책이 없을 것이고요. 전체적으로 책이 중간 이상으로 넘어가면서 첫 부분의 컨셉을 지키지 못다는 점이 아쉬웠습니다.

아쉬운 점은 아닙니다만 표지도 호불호가 갈리는 이유 중 하나가 될 것 같습니다. 비슷한 느낌으로 성안당에서 나오는 '만화로 배우는 ~' 시리즈가 있는데요. 좋은 책임에도 표지 때문에 꺼려하시는 분들이 있는 것으로 알고 있습니다. 이 책도 비스무리한 느낌이 있는데요. 이런 이유 때문에 책 구매를 거부하시는 분들도 몇 있지 않을까 생각합니다. :) 