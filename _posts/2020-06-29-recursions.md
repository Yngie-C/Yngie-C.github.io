---
layout: post
title: 재귀와 병합 정렬 (Recursion & Merge Sort)
category: Data Structure
tag: Data Structure
---



본 게시물은 [카이스트 문일철 교수님의 강의](https://www.edwith.org/datastructure-2019s/lecture/40291/) 를 참조하여 작성하였습니다.



# Recursions

## Repeating Problems

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Mandelbrot_sequence_new.gif/200px-Mandelbrot_sequence_new.gif" alt="fractal" style="zoom:150%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/Fractal">wikipedia : Fractal</a></p>

어떤 문제들은 위 그림에서 볼 수 있는 것처럼 작게 나누더라도 동일한 구조를 가지는 경우가 있다. 회사의 예산을 나누는 경우를 생각해보자. 정해진 총 예산 $\mathbf{N}$ 을 각 부서에서 $N_1, N_2, \cdots$ 로 나눠 가지며 각 부서마다 할당된 예산 $N_n$ 은 부서 내부의 각 팀에 $n_1, n_2,\cdots$ 만큼 할당된다.

문제를 더 작게 나눌수록 문제의 크기는 작아지지만 구조적으로는 동일함을 알 수 있다. 이렇게 쪼개도 원래의 문제와 동일한 구조를 가지는(Self-similar) 문제를 **Repeating Problem** 이라고 하며 나누는 행위를 **Divide** , 그리고 나눈 문제를 해결하는 것을 **Conquer** 라고 한다. 아래는 큰 인형을 분해하면 같은 모습의 더 작은 인형이 등장하는 러시아 전통인형 마트료시카다.

<p align="center"><img src="https://www.dookinternational.com/blog/wp-content/uploads/2018/07/a2.jpeg" alt="Matryoshka" style="zoom: 80%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.dookinternational.com/blog/russian-doll-centuries-old-tradition-and-masterart/">dookinternational.com</a></p>

이러한 Repeating Problem에 해당하는 문제는 어떤 것들이 있을까? 대표적으로 수학에서 점화식(Mathmatical Induction)으로 나타나는 것들이 Repeating Problem의 형태를 가진다. 예를 들어 팩토리얼(Factorial)을 구하는 과정을 수식으로 나타내보자.


$$
\text{Factorial}(n) = \begin{cases} 1 \qquad \qquad \qquad \qquad \qquad \text{if} \quad n = 0\\ n \times n-1 \times \cdots \times 2 \times 1 \quad \text{if} \quad n > 0 \end{cases} \\
\text{Factorial}(n) = \begin{cases} 1 \qquad \qquad \qquad \qquad \qquad \text{if} \quad n = 0\\ n \times\text{Factorial}(n-1) \qquad \text{if} \quad n > 0 \end{cases}
$$



이외에도 최대 공약수를 찾는 유클리드 알고리즘도 사이즈가 줄어들면서도 계속 같은 함수를 호출하기 때문에 Repeating Problem의 형태를 가지는 하나의 예시라고 할 수 있다.



## Recursion

**재귀(Recursion)** 를 사용하는 함수를 파이썬 코드로 구현하면 일반적으로 다음과 같은 형태가 된다.

```python
def recursionFunction(target):
    # ... 함수 내용
    recursionFunction(target`) # 재귀 호출
    # ... 함수 내용
    if escapeCondition:	# 탈출을 위한 조건문
        return Value
```

피보나치 수열(0,1 에서 시작하여 )을 구현한 함수를 예를 들어보도록 하자. 

```python
def Fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    intRet = Fibonacci(n-1) + Fibonacci(n-2)
    return intRet
```

`Fibonacci(4)` 일때 이 함수가 실행되는 과정을 나타내면 아래 그림과 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/85964227-c4502680-b9f3-11ea-82c0-75d65d745982.png" alt="fibo_flow" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

먼저 F(4)로부터 F(3)과 F(2)가 호출된다. F(3)은 다시 F(2)와 F(1)을 호출하며 F(2)는 F(1)과 F(0)을 호출하게 된다. 맨 아래 있는 F(1), F(0)은 탈출문에 의해 리턴값을 반환하게 되며 이 리턴값으로부터 다시 거슬러 올라가 F(4)값을 구할 수 있게 된다.

다음은 동일한 예시를 바탕으로 이런 재귀 함수를 호출했을 때 컴퓨터 안에서 어떤 일이 일어나는지를 살펴보자. 재귀 함수의 호출은 **스택 프레임(Stack Frame)** 에 Item을 쌓아가는 과정이다. 스택 프레임이란 함수 호출 이력을 저장하는 스택이며 함수가 호출되면 Push되고, 함수가 끝나거나 리턴되면 Pop이 된다. 저장되는 Item에는 함수 내 **지역 변수(Local Variable)**와 **함수 호출 파라미터(Function call parameter)**가 있다. 아래의 그림은 F(4)가 호출되는 과정을 스택프레임으로 도식화한 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/85965825-f2843500-b9f8-11ea-81d2-8b152f34fa6d.png
" alt="stackframe" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

위 이미지에서 R.A란 Return Address의 줄임말로 주소 속에 어떤 함수가 호출되었는지를 기록하는 역할을 한다. 위 그림을 보면 맨 먼저 호출된 F(4)가 스택 프레임에 Push된다. 이후는 그래프의 가장 왼쪽 가지만을 따라가 보자. F(3)이 Push되고 그로부터 F(2)가 Push된다. 그러고 나면 F(1)이 Push되고난 후 Return되면서 Pop되고, 다시 F(0)이 Push되어 스택에 들어온 이후에 Pop되는 과정을 거친다. 



## 병합 정렬(Merge Sort)

재귀의 대표적인 예시인 병합 정렬에 대해서 알아보자. 병합 정렬은 크게 Decomposition과 Aggregation의 두 가지 메커니즘으로 나뉘어져 있다. Decomposition은 하나의 리스트를 쪼개는 과정이며 이 과정에서 동일한 함수를 반복적으로 호출하는 Recursion이 일어난다. 리스트 내 모든 요소가 쪼개지면 이를 정렬하면서 병합하는 Aggregation 과정이 진행된다.  아래 그림은 리스트 *[3, 8, 4, 2, 1, 6, 7, 5]* 이 병합 정렬을 통해 정렬되는 과정이다. 



<p align="center"><img src="https://user-images.githubusercontent.com/45377884/85965861-16477b00-b9f9-11ea-86af-ce3d365ea639.png" alt="merge_sort" style="zoom:67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

위 이미지를 파이썬 코드로 구현하면 다음과 같이 쓸 수 있다.

```python
import random

def performMergeSort(lstElementToSort):
    # 탈출을 위한 조건문
    if len(lstElementToSort) == 1:
        return lstElementToSort

    """
    Decomposition
    1개를 길이가 같은 2개의 리스트로 분리한다
    """
    lstSubElementToSort1 = []
    lstSubElementToSort2 = []

    for itr in range(len(lstElementToSort)):
        if len(lstElementToSort)/2 > itr:
            lstElementToSort1.append(lstElementToSort[itr])
        else:
            lstElementToSort2.append(lstElementToSort[itr])
	
    """
    Recursion
    리스트의 원소가 1개가 될 때까지 분리 과정을 반복한다
    """
    lstSubElementToSort1 = performMergeSort(lstSubElementToSort1)
    lstSubElementToSort2 = performMergeSort(lstSubElementToSort2)

    """
    Aggregation
    각 리스트 앞부분 부터 요소의 크기를 순차적으로 비교한 뒤
    크기가 작은 것부터 새로운 리스트의 앞부분에 배치한다
    """
    idxCount1 = 0
    idxCount2 = 0

    for itr in range(len(lstElementToSort)):
        if idxCount1 == len(lstSubElementToSort1):
            lstElementToSort[itr] = lstSubElementToSort2[idxCount2]
            idxCount2 += 1
        elif idxCount2 == len(lstSubElementToSort2):
            lstElementToSort[itr] = lstSubElementToSort1[idxCount1]
            idxCount1 += 1
        elif lstSubElementToSort1[idxCount1] > lstSubElementToSort2[idxCount2]:
            lstElementToSort[itr] = lstSubElementToSort2[idxCount2]
            idxCount2 += 1
        else:
            lstElementToSort[itr] = lstSubElementToSort1[idxCount1]
            idxCount1 += 1
    return lstElementToSort
```

 

## 문제점

재귀 호출에도 문제점이 있다. 함수를 반복해서 호출하기 때문에 함수 호출 횟수가 너무 많아진다는 점이다. 위에서 살펴본 피보나치 함수 F(4)를 표현한 그래프를 다시 보자.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/85964227-c4502680-b9f3-11ea-82c0-75d65d745982.png" alt="fibo_flow" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

F(4) 임에도 함수 호출이 8번이나 이루어진 것을 알 수 있다. F(n) 에서 n의 크기가 더 커져 버린다면 함수를 호출하는 횟수는 훨씬 더 커질 것이다. 이렇게 되면 컴퓨터의 시공간적 효율성이 너무 떨어진다. 이런 문제가 발생하는 원인은 최하단에 위치한 탈출 조건문까지 가서야 스택 프레임에서 빠져나올 수 있다는 데 있다. 하지만 한 번 구한 F(2)를 어디엔가 저장해 놓았다면 오른쪽 노드에서 F(2)를 구할 때는 2번 더 호출하는 일이 없을 것이다. 이런 방식으로 재귀의 함수 호출 문제를 풀기 위해 등장하는 것이 바로 **동적 프로그래밍(Dynamic Programming)** 이다. 