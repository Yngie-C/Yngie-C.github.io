---
layout: post
title: 알고리즘 분석 (Algorithm Analysis)
category: Data Structure
tag: Data Structure
---



본 게시물은 [문일철 교수님의 데이터 구조 및 분석](https://www.edwith.org/datastructure-2019s/lecture/40291/)을 참조하여 작성하였습니다.



# 알고리즘 분석하기

**알고리즘(Algorithm)** 이란 문제를 풀기위해 따라야 하는 일련의 지시사항이다. 알고리즘에는 입력값과 출력값이 존재한다. **자료구조(Data structure)** 는 자료를 조직하는 방법이다. 많은 프로그램이 일련의 알고리즘과 그에 맞는 자료구조를 조합하여 구현되었다. 

이전에 등장했던 삽입, 삭제, 탐색과 같은 Operation이 모두 알고리즘의 범주 내에 들어가 있다. 그 중 정렬(Sorting)은 자료구조 내에 존재하는 요소들을 일정한 조건을 만족하도록 배치하는 알고리즘이다. 정렬 알고리즘은 버블 정렬(Bubble sort), 퀵 정렬(Quick sort) 등 다양하게 구성되어 있다. 그 중 하나인 Merge sort에 대해서는 지난 [재귀 파트]([https://yngie-c.github.io/data%20structure/2020/06/29/recursions/](https://yngie-c.github.io/data structure/2020/06/29/recursions/)) 에서 배우기도 했다.



## 버블 정렬

뒤에 이어질 이야기에서 사용하기 위하여 **버블 정렬(Bubble sort)** 에 대해 알아보자. 버블 정렬은 수많은 정렬 알고리즘 중 매우 직관적인 편에 속하는 정렬이다. 중첩된 For문(Nested for loop)을 통해서 모든 두 요소의 조합을 만들어 비교한다. 오름차순으로 정렬하는 알고리즘이면, 앞에 기준이 되는 숫자가 뒤보다 더 클 경우 둘의 순서를 바꾸어 준다. 이 과정을 계속 반복하면 정렬이 완료된다. 아래는 오름차순 버블 정렬을 파이썬 코드로 구현한 것이다. 

```python
def bubbleSort(lst):
    for itr1 in range(0, len(lst)):
        for itr2 in range(itr1+1, len(lst)):
            if lst[itr1] > lst[itr2]:
                lst[itr1], lst[itr2] = lst[itr2], lst[itr1]

    return lst
```

리스트의 요소가 10개일 때, 이 코드에서 요소를 비교하는 연산이 총 몇 번이나 되풀이 되는지를 계산해보자. 먼저 itr1이 0일 때는 itr2가 1부터 9까지 9번의 비교를 되풀이한다. 그 다음에는 itr1이 1로 증가하고 itr2는 2부터 9까지 8번의 비교를 되풀이한다. 이 과정을 itr1이 9가 될 때까지 반복하게 되는데 각 단계마다 진행되는 비교 횟수는 9, 8, 7, 6, 5, 4, 3, 2, 1번이다. 이를 모두 더하면 10개의 요소를 갖는 리스트에서 버블 정렬을 시도하면 총 45회의 비교가 일어남을 알 수 있다.

리스트 요소의 개수가 $n$ 개로 늘어나면 어떻게 될까? 각 단계마다 $n-1, n-2, \cdots , 1$ 번의 비교가 일어나므로 이를 모두 더하면 된다. 이를 수식으로 나타내면 다음과 같다.


$$
(n-1)+(n-2)+\cdots+1 = \frac{n(n-1)}{2} = \frac{1}{2}n^2 - \frac{1}{2}n
$$


사실 버블 정렬 알고리즘은 다른 정렬 알고리즘과 비교했을 때 매우 비효율적인 알고리즘이다. 10,000명의 회원에 대해 회원 정보를 버블 정렬을 사용하여 오름차순 정렬 한다고 해보자. 이 경우 약 50,000,000번의 연산을 해야 하는데 한 번의 비교를 하는데 0.00001초가 걸리는 컴퓨터를 사용한다면 500초, 즉 8분이 넘는 시간이 걸리게 된다. (물론 컴퓨터의 성능은 더 좋겠지만) 수십만의 고객을 가지고 있는 서비스가 정렬 한 번에 수십분이 넘는 시간을 기다려야 한다면 이는 업무에 방해가 되는 비효율적인 요소이다.



## 알고리즘 분석이란

알고리즘 분석은 각 알고리즘이 요구하는 자원을 추측하는 것이다. 이 자원에는 메모리, 네트워크의 대역폭(bandwidth), 계산 시간(Computational time)이 있다. 메모리와 네트워크의 대역폭은 늘릴 수 있지만 특정 알고리즘의 계산 시간은 줄일 수 없기 때문에 우리는 계산 시간을 가장 단축할 수 있도록 프로그램을 설계해야 한다.

프로그램이 동작하는 시간에 영향을 주는 요소들은 다양하다. 물론 위에서도 언급한 것처럼 좋은 컴퓨터를 사용하면 시간을 단축할 수 있다. 하지만 이외에도 효율적인 자료구조를 선택하고 그에 맞는 알고리즘을 설계하는 것이 더욱 중요하다. 같은 알고리즘과 자료구조를 사용하더라도 입력 데이터가 어떻게 들어오는지에 시간이 바뀌기도 한다. 예를 들어, 이진 탐색 트리에서 평균적인 탐색 시간은 $O(\log n)$ 이다. 하지만 자료가 계속 커지는(혹은 작아지는) 순서로 삽입되어 트리임에도 마치 Linked List와 같이 구성되면 평균적인 탐색 시간이 $O(N)$ 으로 늘어난다.

우리는 사용자가 어떤 방식으로 연산을 실행할 지 알지 못한다. 다시 말해서, 무슨 짓을 할 지 모르기 때문에 최악의 경우(worst-case)를 상정하고 각 알고리즘이 얼마나 효율적인지를 추측해야 한다.



## Big-O Notation

**빅-오 표기법(Big-O Notation)** 이란 알고리즘이 최악의 경우를 만났을 때 걸리는 시간을 표기하는 방식이다. 만약 알고리즘 내에 `if` 조건문이 있다면 모든 조건문을 다 실행한다고 가정한다. 버블 정렬의 경우를 다시 보자.

```python
def bubbleSort(lst):
    for itr1 in range(0, len(lst)):		#Line 1
        for itr2 in range(itr1+1, len(lst)): #Line2 - 4
            if lst[itr1] > lst[itr2]:
                lst[itr1], lst[itr2] = lst[itr2], lst[itr1]

    return lst #Line 5
```

먼저 Line1에서 반복되는 횟수는 $N$ 이다. Line 2,3,4 에서는 각각 $N(N-1)/2$ 번의 반복이 일어난다. 물론 요소 조합에 따라 `if`  문을 건너뛰는 경우가 있지만 최악의 경우를 알아야 하므로 모든 `if` 문이 다 실행된다고 가정한다. 그리고 마지막 Line5에서 값을 반환하면서 $1$ 번의 연산이 일어난다. 버블 정렬에서 일어나는 총 연산의 수를 수식으로 나타내면 아래와 같다.


$$
N + 3 \times \frac{N(N-1)}{2} + 1 = \frac{3}{2}N^2 - \frac{1}{2}N + 1
$$


빅-오 표기법을 나타내는 방법은 다음과 같다. 알고리즘의 연산 횟수를 $f(N)$ 라 하고 그에 맞는 빅-오 표기법을 $O(g(N))$ 라 하면 $N \geq n_0$ 인 $N$ 에 대하여 $f(N) \leq c \cdot g(N)$ 을 만족시키는 양의 상수 $c, n_0$ 가 있어야 한다.

위에서 구한 버블 정렬의 연산 반복 수는 $f(N) = \frac{3}{2}N^2 - \frac{1}{2}N + 1$ 로 나타났다. 이에 해당하는 $g(N) = N^2$ 이라고 해보자. $c$ 에 적당한 값인 $5/2$ 를 대입하고 $f(N) \leq c \cdot g(N)$ 식을 정리하면 다음과 같다. ( $c$ 에 다른 수를 입력해도 된다.)


$$
f(N) \leq c \cdot g(N) \\
\frac{3}{2}N^2 - \frac{1}{2}N + 1 \leq \frac{5}{2} \cdot N^2 \\
N^2 + \frac{1}{2}N -1 \geq 0
$$


마지막 식을 그래프로 나타내면 아래와 같이 나타낼 수 있다. 아래와 같은 포물선 그래프에서 오른쪽 근은 약 $0.781$ 정도의 값을 가진다. 따라서 $n_0 = 0.781$ 이라고 하면 $N \geq n_0$ 인 모든 $N$ 에 대하여 $f(N) \leq c \cdot g(N)$ 을 만족하므로 $O(N^2)$ 는 버블 정렬 알고리즘을 빅-오 표기법으로 잘 나타낸 것이라 할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/86917439-0ac62380-c160-11ea-9524-4eb45c95673f.png" alt="bubblesort_graph" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.wolframalpha.com/">wolframalpha.com</a></p>

$g(N)$ 의 차수가 더 작아지면 어떻게 될까? $g(N) = N$ 이라고 하면 $f(N) \leq c \cdot g(N)$ 을 다음과 같이 정리할 수 있다.



$$
\frac{3}{2}N^2 - \frac{1}{2}N + 1 \leq c \cdot N \\
-\frac{3}{2}N^2 + (c-\frac{1}{2})N -1 \geq 0
$$


맨 아래 있는 식은 위로 볼록한 포물선이 되므로 언젠가는 0보다 작아지게 되고, 이 경우에는 $N \geq n_0$ 인 모든 $N$ 에 대해 식을 만족하는 $n_0$ 를 찾을 수 없다. 따라서 $O(N)$ 은 버블 정렬의 빅-오 표기법으로 적절하지 않다.

반대로 $g(N)$ 의 차수가 더 커지면 어떻게 될까? $g(N) = N^3$ 이라고 하면 $f(N) \leq c \cdot g(N)$ 을 다음과 같이 정리할 수 있다.



$$
\frac{3}{2}N^2 - \frac{1}{2}N + 1 \leq c \cdot N^3 \\
c \cdot N^3-\frac{3}{2}N^2 + \frac{1}{2}N -1 \geq 0
$$


$c$ 는 임의의 양의 상수이므로 마지막 식은 우상향 하는 삼차함수의 식을 그리게 된다. $c=1$ 로 가정했을 때의 그래프는 다음과 같다. 아래와 같은그래프에서 근은 약 $1.584$ 정도의 값을 가진다. 따라서 $n_0 = 1.584$ 이라고 하면 $N \geq n_0$ 인 모든 $N$ 에 대하여 $f(N) \leq c \cdot g(N)$ 을 만족하므로 $O(N^3)$ 는 버블 정렬 알고리즘을 빅-오 표기법으로 잘 나타낸 것이라 할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/86918571-deaba200-c161-11ea-9f8b-06bf63f3c464.png" alt="bubblesort_graph2" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.wolframalpha.com/">wolframalpha.com</a></p>

결론적으로 $g(N)$은 $f(N)$ 의 Upper bound 이다. 그렇다면 특정 $f(N)$ 에 대해 $g(N)$ 의 개수는 매우 많아질 수 있다. $N^k$ 와 같은 다항함수 말고도 $k^N$ 과 같은 지수함수까지 있기 때문이다. 이 많은 $g(N)$ 중 가장 작은 경우, 즉 Tight Upper bound의 경우가 가장 좋은 답이며 일반적으로도 이 경우가 사용된다. 즉 버블 정렬의 빅-오 표기법으로 가장 올바른 것은 $O(N^2)$ 가 된다.

아래는 여러 $g(N)$ 에 대한 Growth Rate를 그래프로 나타낸 것이다.

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Comparison_computational_complexity.svg/1024px-Comparison_computational_complexity.svg.png" alt="growth_rate" style="zoom:50%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://en.wikipedia.org/wiki/Time_complexity">wikipedia - Time Complexity</a></p>

그래프에서도 볼 수 있듯 각 Growth Rate Order는 다음과 같이 나타낼 수 있다. 아래 식에서 $C \geq 2 , k > 2$ 이다.


$$
C^N > N^k > N^2 > N \log N > N > \log N > C
$$

여러 알고리즘이 결합된 형태의 빅-오 표기법은 아래와 같다. 시간 알고리즘이 $f_1(N) = O(g(N)), f_2(N) = O(h(N))$ 와 같은 두 알고리즘에 대하여 다음과 같은 규칙을 만족한다.


$$
f_1(N) + f_2(N) = \max(O(g(N)),O(h(N))) \\
f_1(N) \cdot f_2(N) = O(g(N))*O(h(N))
$$


다음은 List, Stack, Queue 자료구조에 대하여 각 Operation을 빅-오 표기법으로 나타낸 것이다.

|         | List   | Stack  | Queue  |
| ------- | ------ | ------ | ------ |
| Pop     | X      | $O(1)$ | X      |
| Push    | X      | $O(1)$ | X      |
| Enqueue | X      | X      | $O(1)$ |
| Dequeue | X      | X      | $O(1)$ |
| Search  | $O(N)$ | X      | X      |

다음은 Linked List, 일반적인 이진 탐색 트리, 이진 탐색 트리의 가장 좋지 않은 경우에 대해서 각 Operation을 빅-오 표기법으로 나타낸 것이다.

|                     | Linked List | Binary Search Tree in Average | Binary Search Tree in Worst Case |
| ------------------- | ----------- | ----------------------------- | -------------------------------- |
| Search              | $O(n)$      | $O(\log n)$                   | $O(n)$                           |
| Insert after search | $O(1)$      | $O(1)$                        | $O(1)$                           |
| Delete after search | $O(1)$      | $O(1)$                        | $O(1)$                           |
| Traverse            | $O(n)$      | $O(n)$                        | $O(n)$                           |

빅-오 표기법 이외에도 빅-세타(Big - $\Theta$ ) 표기법, 스몰-오(Small-o) 표기법, 스몰-세타(Small - $\theta$ ) 표기법 등이 있으나, 실제로 가장 중요한 것이 최악의 경우에 대한 시간 복잡도를 구하는 것이기 때문에 일반적으로 빅-오 표기법을 가장 많이 사용한다.