---
layout: post
title: 스택(Stack) & 큐(Queue)
category: Data Structure
tag: Data Structure
---



본 게시물은 [카이스트 문일철 교수님의 강의](https://www.edwith.org/datastructure-2019s/lecture/40291/) 를 참조하여 작성하였습니다.



# 스택 (Stack)

이번에는 Singly Linked List의 변형인 **스택(Stack)** 과 **큐(Queue)** 에 대해 알아보자. 먼저 스택을 알아보도록 하자. 스택은 **통로가 한 곳으로 지정**된 경우에 사용되는 자료구조이다. 예를 들어 비행기에 짐을 싣는 경우를 생각해보자. 비행기에서 짐이 들어가는 통로는 한 곳 뿐이므로 맨 마지막으로 들어갔던 짐부터 차례대로 나오게 되며, 처음 들어간 짐은 맨 마지막에 꺼내지게 된다.



## 스택의 구조

스택은 Linked List와 비슷한 구조를 가지고 있다. 차이점은 Linked List에서는 중간에 위치한 인스턴스에 접근할 수 있었지만 스택은 첫 번째 인스턴스인 Top을 통해서만 데이터를 넣고 뺄 수 있도록 한다는 점이다. 아래 그림과 같이 6개의 인스턴스가 형성되려면 가장 먼저 'Cargo Object 6'을 넣은 뒤에 'Cargo Object 5'를 넣고, 이런 방식을 'Cargo Object 1'이 Top이 될 때까지 반복해야 한다. 또는 이 스택에서 데이터를 제거하고자 한다면 Top인 'Cargo Object 1'부터 제거할 수 있다. 이런 후입선출식 메커니즘을 **Last-In-First-Out(LIFO)** 라고 부른다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/85473504-4d2f2280-b5ee-11ea-8e13-103cae501a71.png" alt="stack" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

## 스택의 Operation

스택은 Top을 통해 삽입하고 삭제하는 두 가지 Operation이 있다. 삽입하는 것을 Push라고 하며, 삭제하는 것을 Pop이라고 한다. Push는 Linked List의 첫 번째 인스턴스에 삽입하는 것과 동일하며, Pop은 Linked List의 첫 번째 인스턴스를 삭제하는 것과 동일하다.

아래는 Singly Linked List를 활용하여 스택을 구현하는 파이썬 코드이다.

```python
import SinglyLinkedList

class Stack(object):
    lstInstance = SinglyLinkedList()

    def push(self, value):
        self.lstInstance.insertAt(value, 0)
    def pop(self):
        return self.lstInstance.removeAt(0)
```



# 큐(Queue)

큐는 스택과는 다르게 **통로가 2개인 경우**, 즉 데이터가 들어가는 곳과 나가는 곳이 다른 경우에 사용되는 자료구조이다. 공항에서 비행기를 타기 위해 서있는 줄서있는 사람들을 떠올려보자. (굳이 공항이 아니라도 줄서있는 모든 경우를 생각해도 된다.) 이 때는 먼저 줄을 선 사람이 먼저 일을 해결하고 들어왔던 곳하고는 다른 통로로 나가게 된다. 컨베이어 벨트도 큐의 방식을 사용하고 있다. 먼저 들어온 물건이 작업을 수행하고 다른 통로로 빠져나가고 뒤이어 들어온 물건들은 먼저 나간 물건을 따라 순서대로 나가게 된다.



## 큐의 구조

큐 또한 Linked List와 비슷한 구조를 가지고 있다. 차이점은 Linked List에서는 중간에 위치한 인스턴스에 접근할 수 있었지만 큐는 데이터를 마지막 인스턴스를 통해서만 넣을 수 있고, 첫 번째 인스턴스를 통해서만 뺄 수 있도록 한다는 점이다. 아래 그림과 같이 4개의 인스턴스로 구성된 큐가 완성되려면 가장 먼저 'Product 1'을 넣은 뒤 'Product 2'를 넣는 과정을 반복해야 한다. 이 큐에서 데이터를 제거할 때는 맨 앞에 위치한 'Product 1'부터 제거된다. 이런 선입선출식 메커니즘을 **First-In-First-Out(FIFO)** 라고 부른다.

![queue](https://user-images.githubusercontent.com/45377884/85473621-79e33a00-b5ee-11ea-8e0d-70db0b38f012.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

## 큐의 Operation

큐에도 스택과 같이 두 가지 Operation이 있다. 마지막 인스턴스를 통해 삽입하는 것을 Enqueue라고 하며 첫 번째 인스턴스를 삭제하는 것을 Dequeue라고 한다. 

아래는 Singly Linked List를 활용하여 큐를 구현하는 파이썬 코드이다.



```python
import SinglyLinkedList

class Queue(object):
    lstInstance = SinglyLinkedList()

    def Enqueue(self, value):
        self.lstInstance.insertAt(value, self.lstInstance.getSize())
    def dequeue(self):
        return self.lstInstance.removeAt(0)
```

