---
layout: post
title: Linked List
category: Data Structure
tag: Machine Learning
---



본 게시물은 [카이스트 문일철 교수님의 강의](https://www.edwith.org/datastructure-2019s/lecture/40291/) 를 참조하여 작성하였습니다.



# Linked List

## 기본 구조 : Singly Linked List

이번에는 Linked List에 대해서 알아보자. Linked List는 배열(Array)처럼 인덱스(Index) 구조로 되어있지 않고 노드(Node)와 레퍼런스(Reference)로 이루어져 있다. 노드는 두 가지 변수로 이루어져 있다. 하나는 다음 노드를 가리키는 레퍼런스를 저장하는 변수이고, 다른 하나는 해당 노드에 저장되는 값(Value)을 가리키는 레퍼런스를 저장하는 변수이다.

그리고 일반 노드와는 다른 형태를 가지는 스페셜 노드 Head와 Tail이 있다. 이 두 스페셜 노드는 Singly Linked List를 구성함에 있어서 필수는 아니다. 하지만 이들을 사용하면 검색, 삽입, 삭제를 더 용이하게 할 수 있다. Head는 리스트 맨 앞에 위치하는 노드이다. 이 노드에는 값을 가리키는 레퍼런스는 없으며 오직 다음 노드를 가리키는 변수의 레퍼런스만 있다. Tail은 리스트 맨 마지막에 위치하는 노드이다. 역시 값을 가리키는 레퍼런스는 없으며 다음 노드를 가리키는 변수의 레퍼런스도 없는 노드이다. 오직 다른 노드로부터 레퍼런스를 받기만 하는 노드이다. Head와 Tail은 리스트의 시작과 끝을 담당하는 주춧돌과 같은 역할을 담당한다.



<img src="https://user-images.githubusercontent.com/45377884/85203968-2487fe00-b34c-11ea-80e2-a83b6594d131.png" alt="linkedlist0" style="zoom: 80%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

아래의 코드는 노드를 만드는 과정을 클래스로 구현한 것이다.

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



## 탐색

Linked List의 탐색 절차는 배열과 크게 다르지 않다. 차이점을 찾자면 인덱스를 더 이상 쓸 수 없다는 것이 차이점이다. 먼저 헤드를 찾고 나서, 헤드의 레퍼런스가 가리키는 다음 노드로 이동한다. 그러면 이동한 노드의 레퍼런스가 가리키는 값을 확인한 뒤 맞으면 True를 반환하고, 아니면 다음 노드를 가리키는 레퍼런스를 따라간다. 만약 요소가 리스트 내에 없다면 이 과정을 Tail까지 반복하게 된다. 총 연산 횟수는 배열과 같다. 크기가 N인 리스트에 대하여 리스트 내에 요소가 없을 경우에는 N회, 리스트 내에 요소가 있을 경우에는 최대 N회이다. 



## 삽입

Linked List의 구조가 빛을 발하는 것은 삽입과 삭제 절차이다. 배열에서는 요소를 삽입하거나 삭제하려면 배열 내에 있는 모든 요소를 읽어나가면서 인덱스를 변경해주어야 했다. 하지만 Linked List에서는 노드와 레퍼런스 구조를 활용하기 때문에 이 절차를 단 3번의 조작(Operation)만으로 리스트 내에 요소를 삽입할 수 있다.

1. 우리가 요소를 넣을 자리를 알고 있다는 가정 하에서 시작한다. 요소를 삽입할 위치 다음에 있는 노드 $\text{Node}_{\text{next}}$ 를 미리 저장한다.

2. 그리고 요소를 삽입할 위치 이전에 있는 노드에서 다음 노드를 가리키는 레퍼런스를 새로운 노드 $\text{Node}_{\text{new}}$ 에 덮어씌워 업데이트 해준다.
3. 새로운 노드에서 다음 노드를 가리키는 레퍼런스는 미리 저장해놓은 $\text{Node}_{\text{next}}$ 를 가리키도록 한다.

<img src="https://user-images.githubusercontent.com/45377884/85203979-38cbfb00-b34c-11ea-81fd-3203ffb478e3.png" alt="linkedlist1" style="zoom: 67%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>



## 삭제

Linked List에서는 삭제도 단 3번의 조작만으로 수행할 수 있다. 이 또한 우리가 삭제할 요소의 자리를 알고 있다는 가정 하에서 시작한다.

1. 삭제하는 요소 다음에 위치한 노드 $\text{Node}_{\text{next}}$ 를 검색한다.
2. 삭제하는 이전에 위치한 노드 $\text{Node}_{\text{prev}}$ 에서 다음 노드를 가리키는 레퍼런스가 삭제하는 요소 다음에 위치한 노드를 가리키도록 덮어씌워준다.

이 과정에서 삭제할 요소가 있는 노드 $\text{Node}_{\text{remove}}$ 는 어떤 레퍼런스로부터도 지목받지 못한다. 파이썬에서는 이런 요소를 메모리에서 제거해주는 가비지 콜렉터(Garbage Collecter)를 가지고 있다. 



<img src="https://user-images.githubusercontent.com/45377884/85203980-39fd2800-b34c-11ea-9cf3-46b42626916d.png" alt="linkedlist2" style="zoom: 67%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

## 구현

위에서 살펴본 Linked List에서 요소를 삽입하고 삭제하는 과정을 클래스를 통해 구현해보자. 이전에 선언했던 `Node` 클래스를 활용한다.

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

