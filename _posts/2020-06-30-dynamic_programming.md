---
layout: post
title: 동적 프로그래밍 (Dynamic Programming)
category: Data Structure
tag: Data Structure
---



본 게시물은 [문일철 교수님의 데이터 구조 및 분석](https://www.edwith.org/datastructure-2019s/lecture/40291/)을 참조하여 작성하였습니다.



# Dynamic Programming

재귀(Recursion)으로 구현한 함수에는 한 가지 큰 문제가 있었습니다. 탈출문이 등장할 때까지 문제를 분할(Divide)하기 때문에 함수를 너무 많이 호출하게 된다는 것이었습니다. 이 때문에 굉장히 큰 파라미터를 입력할 경우 스택 프레임에 굉장히 많은 아이템이 쌓이게 되어 시간이 오래 걸리고 굉장히 많은 컴퓨팅 자원을 필요로 했습니다.

이런 문제를 해결하고자 등장한 것이 **동적 프로그래밍(Dynamic programming)**입니다. 동적 프로그래밍은 하위 인스턴스(Sub-Instance)가 반복되는 문제를 해결하기 위해 고안된 알고리즘 설계 기술입니다. 하위 인스턴스는 우리가 구하고자 하는 인스턴스에서 파생되는 인스턴스를 말합니다.

재귀에서 사용했던 예시를 바탕으로 동적 프로그래밍에 대해서 알아보겠습니다. 아래는 재귀로 피보나치를 구현한 후 $fib(7)$을 구할 때 발생하는 함수 호출 그래프입니다.

<img src="https://user-images.githubusercontent.com/45377884/90948208-4bab9a80-e477-11ea-94f7-978c8f08ff7f.png" alt="fib_rec" style="zoom:80%;" />

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.semanticscholar.org/paper/CHAPTER-3-Dynamic-Programming-3-.-1-Fibonacci-Are-Borras/24720d0d6a8869a7674daf3860b2fd463f41a646/figure/0">semanticscholar.org</a></p>

함수 호출 그래프에서 $fib(0), fib(1), fib(2), fib(3)$ 등의 하위 인스턴스가 굉장히 많이 호출되고 있는 것을 볼 수 있습니다. 실제로는 위 이미지에서 초록색으로 색칠된 값만 있어도 나머지 원하는 인스턴스의 값을 구할 수 있는데 말이지요. 동적 프로그래밍은 위와 같이 반복되는 하위 인스턴스를 줄이기 위해 개발되었습니다. 동적 프로그래밍의 기본적인 아이디어는 *'큰 문제를 풀기 위해서 하위 인스턴스 부터 차근차근 해결해보자'* 입니다.

동적 프로그래밍의 스토리라인은 다음과 같습니다.

>우리가 풀어야 할 큰 문제를 그보다 작은 작은 문제외 연결 - 작은 문제를 풀어 구해진 값을 테이블에 저장 - 테이블에 저장된 값중 더 큰 문제를 푸는 데에 필요한 값을 추출하여 사용

## Memoization

**메모이제이션(Memoization)**은 동적 프로그래밍에서 사용되는 기법입니다. 위 스토리라인처럼 작은 문제를 풀어 나온 답을 테이블에 저장하고 한 번 구해진 인스턴스가 다시 구해질 때에는 테이블에 있는 값을 사용합니다.

<img src="https://user-images.githubusercontent.com/45377884/90948377-e5277c00-e478-11ea-9c45-a4cfd4eed001.png" alt="rec_dy"  />

위 그림에서도 알 수 있듯이 재귀는 위에서 아래로 향하는 Top-down 접근 방식이며, 동적 프로그래밍은 Bottom-up이라고 할 수 있습니다. 동적 프로그래밍에서는 한 번 구해진 값에 대해서는 만들어 놓은 테이블에서 값을 꺼내 사용하기 때문에 각 하위인스턴스마다 한 번씩만 값을 구하게 됩니다. 그래서 위 이미지에서 노란색에 해당하는 인스턴스의 값을 구할 필요가 없어지지요.

아래는 피보나치 수열을 동적 프로그래밍으로 구현한 코드입니다.

```python
def FibonacciDP(n):
    """
    Memoization table 구성하기
    딕셔너리로 빈 테이블을 만든 후,
    필요한 최하위 인스턴스를 넣어준다.
    재귀문에서 탈출 조건문에 해당되는 부분을 바꾸어 쓸 수 있다.
    """
    dicFibonacci = {}
    dicFibonacci[0] = 0
    dicFibonacci[1] = 1
	
    """
    더 큰 문제를 풀기 위해서 연결한다.
    """
    for itr in range(2, n+1):
        dicFibonacci[itr] = dicFibonacci[itr-1] + dicFibonacci[itr-2]
    return dicFibonacci[n]
```

동적 프로그래밍 방법을 사용하여 구현한 피보나치 알고리즘의 시간 복잡도를 [빅-O 표기법(Big-O Notation)](https://yngie-c.github.io/data structure/2020/07/08/algorithm_analysis/)으로 나타내면 $O(N)$입니다. 재귀룰 사용하여 구현한 피보나치의 알고리즘의 시간 복잡도가 $O(2^N)$이었던 것을 생각해보면 동적 프로그래밍을 사용하여 구현하는 것이 얼마나 효율적인지를 알 수 있습니다.