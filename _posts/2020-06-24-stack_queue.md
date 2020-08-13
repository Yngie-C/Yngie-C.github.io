---
layout: post
title: 스택(Stack) & 큐(Queue)
category: Data Structure
tag: Data Structure
---



본 게시물은 [카이스트 문일철 교수님의 강의](https://www.edwith.org/datastructure-2019s/lecture/40291/) 를 참조하여 작성하였습니다.



# 스택 (Stack)

이번에는 연결된 리스트(Linked List)의 변형인 스택 과 큐에 대해 알아봅시다. 먼저 스택에 대해 알아보도록 합시다. **스택(Stack)** 은 통로가 한 곳으로 지정된 경우에 사용되는 자료구조입니다. 예를 들어 비행기에 짐을 싣는 경우를 생각해봅시다. 비행기에서 짐이 들어가는 통로는 한 곳 뿐입니다. 그렇기 때문에 맨 마지막으로 들어갔던 짐이 꺼낼 때는 가장 먼저 나오게 되고, 가장 처음에 들어갔던 짐은 가장 나중에 나오게 됩니다.

<p align="center"><img src="https://img.caixin.com/2020-04-07/1586260430322927.jpg" alt="cargo" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.caixinglobal.com/2020-04-07/airlines-convert-passenger-planes-to-haul-cargo-101539686.html">caixinglobal.com</a></p>

## 스택의 구조

스택이 연결된 리스트와 차이를 보이는 점은 첫 번째 노드에만 접근할 수 있다는 점입니다. 연결된 리스트에서 삽입과 삭제했던 방식을 생각해보면 `idxInsert` 와  `idxDelete` 를 통해서 중간에 있는 노드에도 접근할 수 있었습니다. 하지만 스택은 위에서 말한대로 통로가 하나 뿐이므로 맨 위에 있는 노드에만 접근할 수 있습니다. 아래 그림을 보며 설명을 이어 나가겠습니다.

<p align="center"><img src="https://cdn.programiz.com/sites/tutorial2program/files/stack.png" alt="stack"  /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.programiz.com/dsa/stack">programiz.com</a></p>

위 그림은 스택에 요소가 삽입되고 삭제되는 과정을 이미지로 나타낸 것입니다. 빈 스택에 3개의 요소를 삽입하면 위 그림에 2, 3, 4번째 그림같이 쌓이게 됩니다. 이 상황에서 삭제를 시도하면 어떻게 될까요? 통로가 하나밖에 없으니 가장 나중에 들어온 3이 빠져나가게 됩니다. 4번째 그림에서 1을 꺼내고 싶다면 삭제를 총 3번 시도해야 되는 것이지요. 스택에서는 마지막에 들어온 것이 가장 먼저 나가게 되므로 **LIFO(Last-In-First-Out)** 메커니즘을 따른다고 합니다.



## 스택에서의 삽입과 삭제

스택에서는 삽입과 삭제를 각각 `push` 와 `pop` 이라고 부릅니다. 아래는 연결된 리스트를 활용하여 스택에서의 삽입과 삭제를 구현하는 파이썬 코드입니다. `push` 에서는 첫 번째 노드만을 통해서 새로운 노드가 삽입됩니다. `pop` 역시 첫 번째 노드가 삭제되는 것을 알 수 있습니다.

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

큐는 스택과는 달리 통로가 2개인 경우에 사용되는 자료구조 입니다. 통로가 2개이기 때문에 데이터가 삽입되는 통로와 삭제되는 통로가 따로 존재합니다. 공항에서 수하물을 맡기기 위해서 줄을 맞춰 대기하는 경우를 생각해봅시다. 이 경우에는 먼저 줄을 선 사람이 먼저 일을 마치고 줄을 빠져나가게 됩니다. 이때 줄을 서는 방향과 빠져 나가는 방향은 다릅니다.

<p align="center"><img src="https://www.economist.com/img/b/1280/721/90/sites/default/files/images/2015/07/blogs/gulliver/20150801_blp525.jpg" alt="line_up" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.economist.com/gulliver/2015/07/31/line-up-line-up">economist.com</a></p>



## 큐의 구조

큐 또한 연결된 리스트와 유사합니다. 하지만 큐는 통로가 둘 뿐이므로 맨 앞 혹은 맨 뒷 노드에만 접근할 수 있습니다. 아래 그림을 보며 설명을 이어 나가겠습니다.

![queue](https://cdn.programiz.com/sites/tutorial2program/files/queue.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.programiz.com/dsa/queue">programiz.com</a></p>

위 그림에서는 왼쪽에서 데이터가 삽입되고 오른쪽으로 빠져나가고 있습니다. 빈 큐에서 2개의 요소를 삽입하면 3번째 그림과 같이 됩니다. 이 상황에서 요소를 삭제하면 가장 처음에 삽입되었던 1이 꺼내지게 됩니다. 3번째 그림과 같은 큐에서 2를 꺼내고 싶다면 삭제를 2번 실행해주어야 합니다. 큐에서는 먼저 들어온 것일수록 먼저 나가게 되므로 **FIFO(First-In-First-Out)** 메커니즘을 따른다고 합니다.

## 큐에서의 삽입과 삭제

큐에서의 삽입과 삭제도 스택과 같이 특별한 이름을 가지고 있습니다. 큐에서는 삽입과 삭제를 각각 `enqueue` 와 `dequeue` 이라고 부릅니다. 아래는 연결된 리스트를 활용하여 큐에서의 삽입과 삭제를 구현하는 파이썬 코드입니다. `enqueue` 에서는 마지막 노드를 통해서 새로운 노드가 삽입되는 것을 볼 수 있습니다. `dequeue` 는 가장 앞에 있는 요소가 추출되어야 하므로 첫 번째 노드에 접근하여 그 노드를 삭제하는 것을 볼 수 있습니다.

```python
import SinglyLinkedList

class Queue(object):
    lstInstance = SinglyLinkedList()

    def enqueue(self, value):
        self.lstInstance.insertAt(value, self.lstInstance.getSize())
    def dequeue(self):
        return self.lstInstance.removeAt(0)
```

