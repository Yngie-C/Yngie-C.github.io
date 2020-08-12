---
layout: post
title: 연결된 리스트 (Linked List)
category: Data Structure
tag: Data Structure
---



본 게시물은 [카이스트 문일철 교수님의 강의](https://www.edwith.org/datastructure-2019s/lecture/40291/) 를 참조하여 작성하였습니다.



# Linked List

지난 시간에 알아본 [배열(Array)](https://yngie-c.github.io/data structure/2020/06/15/ds_array/) 은 그 자료 구조가 가진 선형 구조 때문에 요소를 삽입과 삭제하는 과정에서 무조건 $N$ 번의 Operation이 필요했습니다. 이번에 알아볼 연결된 리스트(Linked list)는 삽입과 삭제에서 발생하는 비효율을 개선하기 위해서 만들어진 자료구조입니다. 참고로 여기에 등장하는 연결된 리스트와 파이썬에 쓰이는 리스트(List)는 큰 관련이 없으니 다른 자료구조로 생각해주시면 되겠습니다. 



## 연결된 리스트의 기본 구조

**연결된 리스트(Linked List)** 는 노드(Node)와 레퍼런스(Reference)로 되어있습니다. 연결된 리스트가 이런 형식을 택한 이유는 인덱스로 구성된 선형 구조를 피하기 위해서 입니다. 연결된 리스트의 노드는 두 가지 변수로 이루어져 있습니다. 첫 번째는 해당 노드에 저장되는 값을 가리키는 레퍼런스를 저장하는 변수입니다. 그리고 두 번째 변수는 다음 노드를 가리키는 레퍼런스를 저장하는 변수입니다. 아래 그림은 간단한 연결된 리스트의 구조를 이미지로 나타낸 것입니다. 이미지의 연결된 리스트는 총 4개의 노드로 구성되어 있는 것을 볼 수 있습니다.

<img src="https://user-images.githubusercontent.com/45377884/85203968-2487fe00-b34c-11ea-80e2-a83b6594d131.png" alt="linkedlist0" style="zoom: 80%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

연결된 리스트에는 헤드(Head)와 테일(Tail)이라는 스페셜 노드가 있습니다. 위 이미지에서도 이를 확인할 수 있습니다. 헤드는 연결된 리스트 맨 앞에 위치하는 노드이고, 테일은 연결된 리스트 맨 뒤에 위치하는 노드입니다. 두 노드 모두 값을 가리키는 레퍼런스가 없으며, 헤드는 다음 노드를 가리키는 레퍼런스를 가지고 있고 테일은 그마저도 가지고 있지 않습니다. 대신 테일은 다른 노드의 레퍼런스를 받지만 헤드는 다른 노드의 레퍼런스를 받지 않습니다. 이 두 스페셜 노드가 연결된 리스트를 구성하는 데 있어서 필수적인 요소는 아니지만, 연결된 리스트의 주줏돌과 같은 역할을 담당함으로써 삽입과 삭제를 더욱 용이하게 만드는 역할을 합니다.

아래는 연결된 리스트를 만드는 파이썬 코드입니다.

```python
class Node:
    nodeNext = None
    nodePrev = ''
    objValue = ''
    binHead = False
    binTail = False

    def __init__(self, objValue='', nodeNext=None, binHead=False, binTail=False):
        self.nodeNext = nodeNext
        self.objValue = objValue
        self.binHead = binHead
        self.binTail = binTail

    def getValue(self):
        return self.objValue
    def setValue(self, objValue):
        self.objValue = objValue

    def getNext(self):
        return self.nodeNext
    def setNext(self, nodeNext):
        self.nodeNext = nodeNext

    def isHead(self):
        return self.binHead
    def isTail(self):
        return self.binTail


node1 = Node(objValue='a')
nodeTail = Node(binTail=True)
nodeHead = Node(binHead=True, nodeNext=node1)
```



## 연결된 리스트에서의 탐색(Search)

연결된 리스트에서 요소를 탐색하는 과정은 배열과 크게 다르지 않습니다. 연결된 리스트는 배열에서 삽입과 삭제만을 개선하기 위한 자료구조이기 때문이지요. 비록 접근 방법은 다르지만 요소를 하나하나 확인한다는 점에서는 별 차이가 없습니다. 배열에서는 요소를 탐색할 때 인덱스로 접근했지만 연결된 리스트는 인덱스를 사용하지 않기 때문에 노드와 레퍼런스로 접근합니다. 아래 그림을 보며 설명을 이어가겠습니다.

![search_in_ll](https://user-images.githubusercontent.com/45377884/89987576-40a77c00-dcb9-11ea-8f2f-c1de60655858.png)

가장 먼저 헤드를 찾은 뒤 헤드의 레퍼런스가 가리키는 노드로 이동합니다. 그리고 이동한 노드가 참조하는 값을 비교한 뒤 맞으면 거기서 함수를 끝내고 아니면 레퍼런스를 따라 다음 노드로 이동합니다. 이 과정을 찾고자 하는 요소가 나올 때까지 반복하면 됩니다. 만약 요소가 없다면 테일이 나올 때까지 반복한 후 False를 반환하면 됩니다.

연결된 리스트는 탐색 과정이 비슷하기 때문에 발생하는 Operation 횟수 역시 배열과 동일합니다. 내부 요소의 개수가 $N$ 인 연결된 리스트에서 리스트 내에 없는 요소를 탐색하는 경우에는 $N$ 회, 리스트 내에 있는 요소를 탐색하는 경우에는 최대 $N$ 회가 됩니다.



## 연결된 리스트에서의 삽입(Insert)

이제부터 알아볼 삽입과 삭제는 연결된 리스트의 구조가 빛을 발하는 부분입니다. 배열에서 $N$ 회 필요했던 Operation을 연결된 리스트에서는 획기적으로 줄일 수 있습니다. 연결된 리스트에서는 단 $3$ 번의 Operation 만으로 요소를 삽입하고 삭제할 수 있습니다. 먼저 요소를 삽입하는 과정부터 알아봅시다. 배열에서 우리가 요소를 넣을 인덱스를 지정해주었던 것처럼, 연결된 리스트에서도 요소가 들어갈 자리는 미리 지정해 준다는 가정 하에서 시작합니다.

삽입의 첫 번째 절차는 요소를 삽입하려는 위치 다음에 있는 노드인 $\text{Node}_{\text{next}}$ 를 미리 저장하는 것입니다. 다음으로는 노드 삽입할 위치 이전에 있는 노드인 $\text{Node}_{\text{prev}}$ 의 레퍼런스가 가리키는 노드를 $\text{Node}_{\text{next}}$ 에서 새로운 노드인 $\text{Node}_{\text{new}}$ 로 업데이트 해줍니다. 마지막으로 $\text{Node}_{\text{new}}$ 의 레퍼런스가 가리키는 노드를 $\text{Node}_{\text{next}}$ 로 바꿔주면 삽입 절차가 끝나게 됩니다.

연결된 리스트에서는 이 $3$ 번의 Operation 만으로 자료구조 내에 요소를 삽입할 수 있습니다. 정해진 길이없이 노드와 레퍼런스만 조작해주면 되기 때문입니다. 아래는 연결된 리스트에서 요소를 삽입하는 과정을 이미지로 나타낸 것입니다.

<img src="https://user-images.githubusercontent.com/45377884/85203979-38cbfb00-b34c-11ea-81fd-3203ffb478e3.png" alt="linkedlist1" style="zoom: 67%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>



## 연결된 리스트에서의 삭제(Delete)

연결된 리스트에서는 삭제를 단 2번의 Operation 만으로 수행할 수 있습니다. 삭제 역시 우리가 삭제할 요소의 자리를 지정한다는 가정 하에서 시작합니다.

첫 번째 Operation은 삭제하려는 요소의 레퍼런스를 받는 노드인 $\text{Node}_{\text{next}}$ 를 저장하는 것입니다. 이 과정은 삽입에서 했던 것과 동일합니다. 두 번째 Operation은 $\text{Node}_{\text{prev}}$ 의 레퍼런스가 가리키는 노드를 $\text{Node}_{\text{remove}}$ 에서 $\text{Node}_{\text{next}}$ 로 변경해줍니다. 연결된 리스트에서는 두 번의 Operation만을 거치면 삭제 과정이 완료됩니다.

그런데 우리가 다루어주지 않는 한 노드가 있습니다. 바로 $\text{Node}_{\text{remove}}$ 인데요. 이 노드는 이제 어떻게 처리되는 것일까요? 파이썬에는 이런 부분을 처리하는 가비지 콜렉터(Garbage collector)라는 장치가 있습니다. (헤드 노드를 제외한다면) 연결된 리스트에서 어떤 레퍼런스의 지목도 받지 못하는 노드는 바로 이 가비지 콜렉터에 의해서 제거됩니다.

<img src="https://user-images.githubusercontent.com/45377884/85203980-39fd2800-b34c-11ea-9cf3-46b42626916d.png" alt="linkedlist2" style="zoom: 67%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

## 연결된 리스트의 탐색, 삽입, 삭제 구현하기

아래는 연결된 리스트에서 요소를 삽입하고 삭제하는 과정을 파이썬 코드로 구현한 것입니다. 이전에 구현했던 `Node` 클래스를 활용할 수 있습니다.

```python
class SinglyLinkedList:
    nodeHead = ''
    nodeTail = ''
    size = 0

    def __init__(self):
        self.nodeTail = Node(binTail=True)
        self.nodeHead = Node(binTail=True, nodeNext=self.nodeTail)

    def insertAt(self, objInsert, idxInsert):
        nodeNew = Node(objValue=objInsert)
        nodePrev = self.get(idxInsert-1)
        nodeNext = nodePrev.getNext()
        nodePrev.setNext(nodeNew)
        nodeNew.setNext(nodeNext)
        self.size = self.size+1

    def removeAt(self, idxRemove):
        nodePrev = self.get(idxRemove-1)
        nodeRemove = nodePrev.getNext()
        nodeNext = nodeRemove.getNext()
        nodePrev.setNext(nodeNext)
        self.size = self.size-1
        return nodeRemove.getValue()

    def get(self, idxRemove):
        nodeReturn = self.nodeHead
        
        for itr in range(idxRetrieve+1):
            nodeReturn = nodeReturn.getNext()
        return nodeReturn

    def printStatus(self):
        nodeCurrent = self.nodeHead

        while nodeCurrent.getNext().isTail() == False:
            nodeCurrent = nodeCurrent.getNext()
            print(nodeCurrent.getValue(), end=" ")
        print("")

    def getSize(self):
        return self.size
    

list1 = SinglyLinkedList()
list1.insertAt('a',0)
list1.insertAt('b',1)
list1.insertAt('d',2)
list1.insertAt('e',3)
list1.insertAt('f',4)
list1.printStatus()
>>> a b d e f

list1.insertAt('c',2)
list1.printStatus()
>>> a b c d e f

list1.removeAt(3)
list1.printStatus()
>>> a b c e f
```

