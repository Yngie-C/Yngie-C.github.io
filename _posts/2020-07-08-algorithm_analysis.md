---
layout: post
title: 알고리즘 분석 (Algorithm Analysis)
category: Data Structure
tag: Data Structure
---



본 게시물은 [문일철 교수님의 데이터 구조 및 분석](https://www.edwith.org/datastructure-2019s/lecture/40291/)을 참조하여 작성하였습니다.



# Algorithm Analysis

**알고리즘(Algorithm)**이란 무엇일까요? 위키피디아에 나와있는 알고리즘의 정의는 다음과 같습니다.

>*"알고리즘은 수학과 컴퓨터 과학, 언어학 또는 관련 분야에서 어떠한 문제를 해결하기 위해 정해진 일련의 절차나 방법을 공식화한 형태로 표현한 것, 계산을 실행하기 위한 단계적 절차를 의미한다. 알고리즘은 연산, 데이터 진행 또는 자동화된 추론을 수행한다."*

즉 알고리즘은 문제를 푸는 방법에 대한 지시사항입니다. 모든 알고리즘에는 입력값과 출력값이 존재하는데요. 효율적인 알고리즘 구현을 위해서는 그에 맞는 **자료구조(Data structure)**가 동반되어야 합니다. 자료구조는 자료를 조직하는 방법으로 많은 프로그램이 일련의 알고리즘과 그에 맞는 자료구조를 조합하여 구현되어 있습니다. 이전에 각각의 자료구조에 대해 구현했던 삽입(Insert), 삭제(Delete), 탐색(Search)과 같은 작업 모두 알고리즘의 범주 내에 들어갑니다.

이 중에서 **정렬(Sorting)**은 자료구조 내에 존재하는 요소를 일정한 조건을 만족하도록 배치하는 알고리즘입니다. 버블 정렬(Bubble sort), 퀵 정렬(Quick sort)등 다양한 정렬 알고리즘이 있습니다.



## Bubble Sort

이후에 이어질 이야기를 위해서 버블 정렬에 대해 알아보겠습니다. 버블 정렬은 여러 정렬 알고리즘 중에서 매우 직관적인 편에 속하는 정렬입니다. 중첩된 반복문을 통해서 모든 두 요소를 비교합니다. 오름차순으로 정렬하는 알고리즘이라면, 기준이 되는 앞쪽 숫자보다 뒷쪽 숫자가 더 클 경우에 둘의 순서를 바꾸어줍니다. 이 과정을 배열의 끝까지 반복하면 정렬이 완료됩니다. 아래는 버블 정렬을 사용하여 리스트를 오름차순으로 정렬하는 파이썬 코드입니다.

```python
def bubbleSort(lst):
    for itr1 in range(len(lst)):
        for itr2 in range(len(lst)-1):
            if lst[itr2] > lst[itr2+1]:
                lst[itr2], lst[itr2+1] = lst[itr2+1], lst[itr2]

    return lst
```

리스트에 10개의 요소가 있을 때, 위 코드에서 각각의 요소를 몇 번이나 비교할까요? 먼저 `itr1`이 0일 때는 `itr2`가 0부터 8까지 9번의 비교를 되풀이하게 됩니다. `itr2`를 모두 마치고 나면 다시 `itr1`이 1로 증가하게 됩니다. `itr1`이 1일 때에는 0부터 7까지 8번의 비교를 되풀이합니다. 이 과정을 `itr1`이 9가 될 때까지 반복합니다. 각 단계마다 `itr2`가 순환하면서 요소를 비교하는 횟수는 $9, 8, 7, \cdots, 1$ 번입니다. 이를 모두 더하면 요소가 10개인 리스트를 버블 정렬할 때 비교 횟수를 구할 수 있습니다.

이를 요소가 $n$개인 리스트의 버블 정렬로 일반화해 봅시다. 리스트의 요소가 $n$개일 때는 각 단계마다 $n-1, n-2, \cdots , 1$번의 비교가 일어나며 이를 모두 더하면 총 비교 횟수를 구할 수 있습니다. 수식으로 나타내면 다음과 같습니다.


$$
(n-1)+(n-2)+\cdots+1 = \frac{n(n-1)}{2} = \frac{1}{2}n^2 - \frac{1}{2}n
$$


버블 정렬 알고리즘은 다른 정렬 알고리즘과 비교했을 때 매우 비효율적인 알고리즘입니다. 10,000명의 회원에 대한 회원 정보를 버블 정렬을 사용하여 오름차순으로 정렬한다면 몇 번의 반복을 수행해야 할까요? 약 $50,000,000$개의 성분을 비교해야합니다. 만약 모든 요소를 바꿔주어야 한다면 $0.00001$초가 걸리는 컴퓨터를 사용했을 때 500초, 즉 8분이 넘게 걸리게 됩니다(물론 실제로 사용되는 컴퓨터의 성능은 훨씬 좋지만 예를 들어 설명하였습니다).

## Analysis

**알고리즘 분석(Algorithm analysis)**은 알고리즘이 필요로 하는 자원이 얼마나 되는지를 추측하는 것입니다. 이 때 자원에는 메모리, 네트워크의 대역폭(Bandwidth), 계산 시간(Computation time) 등이 모두 포함됩니다. 메모리와 네트워크의 대역폭은 더 늘릴 수 있지만 한 번 설정한 알고리즘의 계산 시간을 줄일 수는 없기 때문에 처음부터 효율적인 계산을 하는 알고리즘을 설계해야 합니다.

같은 알고리즘을 사용하더라도 입력 데이터가 어떤 자료구조의 형태로 들어오는 지에 따라서 계산 시간이 바뀌기도 합니다. 예를 들어, 이진 탐색 트리에서 평균적인 탐색 시간은 $O(\log N)$ 입니다. 그렇다면 이 $O(\log N)$이라는 표기는 무엇을 의미할까요? 이어서 빅-O 표기법에 대해 알아보겠습니다.



## Big-O Notation

**빅-O 표기법(Big-O Notation)**이란 알고리즘이 최악의 경우를 만났을 때 걸리는 시간을 표기하는 방식입니다. 만약 알고리즘 내에 `if` 조건문이 있으면 이 조건을 모두 만족하여 조건문 이하의 코드를 모두 실행한다고 가정하게 됩니다. 버블 정렬의 경우 `if` 조건문을 모두 실행한다면 두 요소를 바꾸는 작업을 $\frac{N(N-1)}{2}$ 번 실행하게 됩니다.

```python
def bubbleSort(lst):
    for itr1 in range(len(lst)): #line1
        for itr2 in range(len(lst)-1): #line2
            if lst[itr2] > lst[itr2+1]: #line3
                lst[itr2], lst[itr2+1] = lst[itr2+1], lst[itr2] #line4

    return lst #line5
```

버블 정렬의 함수 내에서 최악의 경우에 각 코드 Line을 몇 번이나 실행하는지 헤아려 보겠습니다. 먼저 Line1은 몇 번이나 반복될까요? `itr1`이 $0$ 부터 $N-1$ 까지 반복하니 $N$ 번 반복하게 됩니다. Line2는 조건문과 상관없이 $(N-1) + (N-2) + \cdots + 1 = \frac{N(N-1)}{2}$ 번 반복합니다. Line3,4는 조건이 안맞을 경우에는 실행하지 않지만 최악의 경우, 즉 모든 `if`문을 실행하는 경우를 생각하는 것이므로 두 줄 모두 $\frac{N(N-1)}{2}$ 실행됩니다. 마지막 Line5는 함수를 마칠 때 한 번 사용되므로 $1$번 실행하게 됩니다. 이를 모두 합한 $\frac{3}{2}N^2 - \frac{1}{2}N + 1$이 버블 정렬의 반복 연산 횟수가 됩니다.

이제 빅-$O$ 표기법에 대해 본격적으로 알아보겠습니다. 빅-O 표기법을 나타내는 방법은 다음과 같습니다. 우선 테스트할 알고리즘의 연산 횟수를 $f(N)$이라고 합니다. 그기고 그에 맞는 빅-$O$ 표기법이 $O(g(N))$으로 나타난다고 해보겠습니다. 이 때  $N \geq n_0$ 인 $N$ 에 대하여 $f(N) \leq c \cdot g(N)$ 을 만족시키는 양의 상수 $c, n_0$ 가 있어야 합니다.

말로 하면 어려우니 버블 정렬의 예를 들어 설명해보겠습니다. 위에서 구한 버블 정렬의 연산 반복 수 $f(N) = \frac{3}{2}N^2 - \frac{1}{2}N + 1$이었습니다. 이에 해당하는 $g(N) = N^2$으로 가정해보겠습니다. $c$ 에 적당한 값인 $5/2$ 를 대입하고 $f(N) \leq c \cdot g(N)$ 식을 정리하면 다음과 같은 식이 됩니다. ( $c$ 에 다른 수를 대입해도 됩니다.)

$$
f(N) \leq c \cdot g(N) \\
\frac{3}{2}N^2 - \frac{1}{2}N + 1 \leq \frac{5}{2} \cdot N^2 \\
N^2 + \frac{1}{2}N -1 \geq 0
$$


마지막 식을 그래프로 나타내면 아래와 같이 나타낼 수 있습니다. 아래와 같은 포물선 그래프에서 오른쪽 근은 약 $0.781$ 입니다. 따라서 $n_0 = 0.781$ 이라고 하면 $N \geq n_0$ 인 모든 $N$ 에 대하여 $f(N) \leq c \cdot g(N)$을 만족하므로 $O(N^2)$ 는 버블 정렬 알고리즘을 빅-$O$ 표기법으로 잘 나타낸 것이라 할 수 있습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/86917439-0ac62380-c160-11ea-9524-4eb45c95673f.png" alt="bubblesort_graph" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.wolframalpha.com/">wolframalpha.com</a></p>

$g(N)$ 의 차수가 더 작아지면 어떻게 될까요? $g(N) = N$ 이라고 가정하면 $f(N) \leq c \cdot g(N)$을 다음과 같이 정리할 수 있습니다.



$$
\frac{3}{2}N^2 - \frac{1}{2}N + 1 \leq c \cdot N \\
-\frac{3}{2}N^2 + (c-\frac{1}{2})N -1 \geq 0
$$


이 식은 위로 볼록한 포물선이 되므로 언젠가는 0보다 작아지게 됩니다. 이 경우에는 $N \geq n_0$ 인 모든 $N$ 에 대해 $f(N) \leq c \cdot g(N)$을 만족하는 $n_0$ 를 찾을 수 없습니다. 따라서 $O(N)$ 은 버블 정렬의 빅-$O$ 표기법으로 적절하지 않습니다.

반대로 $g(N)$ 의 차수가 더 커지면 어떻게 될까요? $g(N) = N^3$ 이라고 하면 $f(N) \leq c \cdot g(N)$ 을 다음과 같이 정리할 수 있습니다.



$$
\frac{3}{2}N^2 - \frac{1}{2}N + 1 \leq c \cdot N^3 \\
c \cdot N^3-\frac{3}{2}N^2 + \frac{1}{2}N -1 \geq 0
$$


$c$ 는 임의의 양의 상수이므로 마지막 식은 우상향 하는 삼차함수의 식을 그리게 된다. $c=1$ 로 가정했을 때의 그래프는 다음과 같습니다. 아래와 같은 그래프에서 근은 약 $1.584$ 입니다. $n_0 = 1.584$ 이라고 하면 $N \geq n_0$ 인 모든 $N$ 에 대하여 $f(N) \leq c \cdot g(N)$ 을 만족합니다. 따라서 $O(N^3)$ 는 버블 정렬 알고리즘을 빅-$O$ 표기법으로 잘 나타낸 것이라 할 수 있습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/86918571-deaba200-c161-11ea-9f8b-06bf63f3c464.png" alt="bubblesort_graph2" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.wolframalpha.com/">wolframalpha.com</a></p>

결론적으로 $g(N)$은 $f(N)$ 의 상한선(Upper bound)이 되는 함수입니다. 그렇다면 특정 $f(N)$ 에 대해 $g(N)$ 의 개수는 매우 많아질 수 있습니다. $N^k$ 와 같은 다항함수 말고도 $k^N$ 과 같은 지수함수까지 있기 때문입니다. 이 많은 $g(N)$ 중 가장 작은 경우, 즉 Tight한 상한선의 경우가 가장 좋은 답이며 일반적으로도 이 경우가 사용됩니다. 즉 버블 정렬의 빅-$O$ 표기법으로 가장 올바른 것은 $O(N^2)$ 가 됩니다.

## Growth rate

아래는 여러 $g(N)$에 대한 성장률(Growth rate)을 그래프로 나타낸 것입니다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Comparison_computational_complexity.svg/1024px-Comparison_computational_complexity.svg.png" alt="growth_rate" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/Time_complexity">wikipedia - Time Complexity</a></p>

그래프에서도 볼 수 있듯 각 성장률 순서(Growth rate order)는 다음과 같이 나타낼 수 있습니다.


$$
N! > C^N > N^k > N^2 > N \log N > N > \log N > C \qquad \text{if, } \quad C \geq 2 , k > 2
$$

여러 알고리즘이 결합된 형태의 빅-$O$ 표기법은 아래와 같습니다. 시간 알고리즘이 $f_1(N) = O(g(N)), f_2(N) = O(h(N))$ 와 같은 두 알고리즘에 대하여 다음과 같은 규칙을 만족합니다.


$$
f_1(N) + f_2(N) = \max(O(g(N)),O(h(N))) \\
f_1(N) \cdot f_2(N) = O(g(N))*O(h(N))
$$

성장률 순서와 결합 규칙을 알면 빅-O 표기법을 쉽게 구할 수 있습니다. 덧셈 결합 규칙을 적용하여 성장률 순서에서 복잡한 순서대로 정렬했을 때, 가장 앞쪽 순서에 위치한 항만 남기면 그것이 $g(N)$이 됩니다. 아래는 $f(N)$을 빅-$O$ 표기법으로 나타내는 몇 가지 예시입니다.

> $f(3N+2 \cdot N^3+3) \Rightarrow O(N^3)$
>
> $f(3N\log N + \log N + 1024) \Rightarrow O(N\log N)$
>
> $f(3N^3 + 2^N) \Rightarrow O(2^N)$
>
> $f(100001) \Rightarrow O(1)$

## Complexity in List, Stack, Queue

다음은 Array(List), Stack, Queue 자료구조에 대하여 삽입(Push, Enqueue), 삭제(Pop, Dequeue) 및 탐색을 빅-$O$ 표기법으로 나타낸 표입니다.

|         | Array(List) | Stack  | Queue  |
| ------- | ----------- | ------ | ------ |
| Push    | X           | $O(1)$ | X      |
| Pop     | X           | $O(1)$ | X      |
| Enqueue | X           | X      | $O(1)$ |
| Dequeue | X           | X      | $O(1)$ |
| Search  | $O(N)$      | X      | X      |

다음은 [연결된 리스트(Linked List)](https://yngie-c.github.io/data structure/2020/06/21/linked_list/), 평균적인 [이진 탐색 트리(Binary search tree)](https://yngie-c.github.io/data structure/2020/07/06/bst/), 이진 탐색 트리의 가장 좋지 않은 경우에 대해서 탐색, 삭제, 트래버스(Traverse)를 빅-$O$ 표기법으로 나타낸 표입니다.

|                     | Linked List | Binary Search Tree in Average | Binary Search Tree in Worst Case |
| ------------------- | ----------- | ----------------------------- | -------------------------------- |
| Search              | $O(n)$      | $O(\log n)$                   | $O(n)$                           |
| Insert after search | $O(1)$      | $O(1)$                        | $O(1)$                           |
| Delete after search | $O(1)$      | $O(1)$                        | $O(1)$                           |
| Traverse            | $O(n)$      | $O(n)$                        | $O(n)$                           |

빅-$O$ 표기법 이외에도 빅-$\Theta$ 표기법 $\Theta(N)$, 스몰-o 표기법, 스몰- $\theta$ 표기법 등이 있습니다. 하지만 가장 중요하게 고려되는 사항은 역시 최악의 경우입니다. 이것만 줄이면 다른 경우는 쉽게 커버할 수 있기 때문이지요. 그러므로 최악의 경우에 대해 시간 복잡도를 구하는 빅-$O$ 표기법이 일반적으로 가장 많이 사용됩니다.