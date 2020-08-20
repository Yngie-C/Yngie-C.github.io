---
layout: post
title: 동적 프로그래밍 (Dynamic Programming)
category: Data Structure
tag: Data Structure
---



본 게시물은 [문일철 교수님의 데이터 구조 및 분석](https://www.edwith.org/datastructure-2019s/lecture/40291/)을 참조하여 작성하였습니다.



# Dynamic Programming

## 정의

동적 프로그래밍(Dynamic programming)은 중복되는 하위 인스턴스(Sub-Instance)가 반복되는 문제를 해결하기 위한 일반적인 알고리즘 설계 기술이다. 하위 인스턴스는 우리가 구하고자 하는 인스턴스에서 파생되는 인스턴스를 말한다. 지난 [재귀]([https://yngie-c.github.io/data%20structure/2020/06/29/recursions/](https://yngie-c.github.io/data structure/2020/06/29/recursions/)) 에서 살펴본 피보나치 수열을 통해서 동적 프로그래밍이 어떻게 작동하는지 알아보도록 하자. 

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/85964227-c4502680-b9f3-11ea-82c0-75d65d745982.png" alt="fibo_flow" style="zoom: 67%;" /></p>

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

위 그림에서 우리가 구하고자 하는 것은 F(4)이지만 이를 구하기 위해서 다양한 인스턴스가 생성된다. 그림에서 F(2)와 같은 하위 인스턴스는 반복되어 호출되는 것을 알 수 있다. 이렇게 발생하는 중복되는 함수 호출을 줄이기 위해서 등장하는 것이 동적 프로그래밍이다.

동적 프로그래밍의 기본적인 아이디어는 '큰 문제를 풀기 위해서 필요한 하위 인스턴스부터 풀어보자'이며 스토리라인은 다음과 같다. 가장 먼저 우리가 풀어야 할 (큰) 문제를 그보다 작은 문제와 연결한다. 그리고 작은 문제를 풀어 구해진 값을 테이블(Table)에 저장한다. 그리고 이 값을 더 큰 문제를 푸는데 추출하여 사용한다.

## Memoization

메모이제이션(Memoization)은 동적 프로그래밍에서 사용되는 중요한 기술이다. 이전에 있었던 함수 호출의 결과를 나중에 재사용할 수 있도록 Table에 저장하는 것이다. 아래 이미지를 통해 재귀와 동적 프로그래밍, 스택 프레임과 메모이제이션(Memoization)을 도식화하여 비교할 수 있다. 

Memoization : 기존의 함수 호출과 그것의 결과를 재활용 하기 위해서 저장하는 것. 스택 프레임과는 상반적인 철학. Memoization은 바텀업 어프로치. 리커젼은 탑다운 어프로치. 접근 방법이 상반된다. 리커전이 오버래핑되는 서브인스턴스가 많으면 다이나믹 프로그래밍을 사용하여 반복되는 함수 호출을 줄여줄 수 있다. 

![dyna_programming_!](https://user-images.githubusercontent.com/45377884/86083641-ce4a5600-bad5-11ea-8cfa-d2b7e7a5fdf4.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

재귀 방식에서는 가장 큰 함수부터 호출하여 스택 프레임에 저장하였다. 그리고 탈출문을 만날 때까지 작은 문제로 나누면서 스택 프레임에 함수 호출을 저장한다. 진행되는 방향 관점에서 재귀(Recursion)는 위에서 아래로 향하는 Top-down 방식이라고 할 수 있다. 동적 프로그래밍은 상반되는 접근 방식을 가지고 있다. 먼저 맨 아래에 위치한 인스턴스부터 접근하여 문제를 해결해 나간다. 그리고 그렇게 산출된 값을 메모이제이션(Memozation)한다. 우리가 원하는 값이 나올 때까지 이 과정을 반복해 나간다.

재귀로 나타냈던 피보나치 수열을 동적 프로그래밍으로 나타내 보자. 동적 프로그래밍을 사용하여 파이썬 코드로 구현한 피보나치 수열은 아래와 같다.

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

이렇게 동적 프로그래밍으로 구현한 피보나치 수열 알고리즘의 시간 복잡도를 빅-오 표기법(Big-O Notation)으로 나타내면 $O(N)$ 이다. 재귀를 사용하여 표현한 경우의 시간 복잡도를 빅-오 표기법으로 나타내면 $O(2^N)$ 이었으나 동적 프로그래밍을 사용하여 이를 획기적으로 줄일 수 있었다.



## 동적 프로그래밍으로 최단 경로 탐색하기 

좀 더 복잡한 경우를 생각해보자. 한 공장에 조립 라인이 2개가 있다고 생각해보자. 라인의 공정 단계(스테이션)마다 하는 일은 동일하지만 기계의 성능이 모두 달라 수행하는 데 걸리는 시간이 다르다고 한다. 현재 각 공정을 처리하는 데 걸리는 시간이 아래 그래프에서 노드에 쓰여진 숫자와 같고 병목 현상이 생길 것을 방지하기 위해서 스테이션 마다 라인을 옮길 수 있다고 하자. 새로운 물건을 어떤 경로로 보내야 가장 빠른지, 그리고 그 시간은 얼마일지는 어떻게 구할 수 있을까?

![assembly_line](https://user-images.githubusercontent.com/45377884/86085745-deb0ff80-bada-11ea-91ca-140bad4d7413.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

위에서 살펴본 바에 의하면 동적 프로그래밍은 문제를 나눈 뒤 가장 단순한 문제부터 풀었다. 위 그래프를 공정 단계마다 나누어 6개로 분리해 보자. 그리고 각 구간까지 수행하는 데 얼마 만큼의 최소 시간이 걸리는지(Min Travel Time), 그리고 이전에 어떤 공정 라인으로부터 오게 되었는지(Retrace)에 대해서 각각 테이블을 만들어 나타내도록 하자.

![assembly_line2](https://user-images.githubusercontent.com/45377884/86086157-fb9a0280-badb-11ea-80e9-8afe12e4166f.png)

<p align="center" style="font-size:80%">이미지 출처 : <a href="https://www.edwith.org/datastructure-2019s/notice/1658">데이터 구조 및 분석 수업자료</a></p>

먼저 첫 번째 보라색 선 이전에 있는 각 라인에 위치한 스테이션에 도달하기 까지는 L1과 L2에서 각각 9와 12의 시간이 걸린다. 그리고 이들은 모두 Start로부터 왔기 때문에 Retrace 테이블에는 S를 기록한다. 두 번째 단계를 보자. L1 스테이션에 도착하기 위한 최소 시간은 9+9=18이다. 물론 L2에서 건너오는 방법도 있지만 12+2+9=23이 되어 최소시간이라고 할 수 없다. 두 번째 단계 L2 스테이션에 도착하기 위한 최소 시간은 9+2+5=16이다. L2에서 그대로 진행하는 방법도 있지만 12+5=17이 되어 시간이 더 오래 걸리게 된다. 이 때, L1과 L2에 위치한 두 스테이션은 이전 단계에 위치한 L1 스테이션으로부터 왔기 때문에 Trace에는 1을 기록한다. 이 과정을 끝까지 반복하면 위와 같이 표를 채울 수 있다.

최종까지 표를 채웠을 때 L1은 35만큼의 시간이, L2는 37만큼의 시간이 걸렸다. 이 스테이션 이후에 End까지 걸리는 시간이 각각 3과 2이다. 따라서 마지막이 L1일 경우 걸리는 시간은 38, L2일 경우 걸리는 시간은 39이므로 새로운 물건을 집어넣을 때 걸리는 최소 시간은 38이라 할 수 있다. 그리고 Retrace 테이블로부터 이 시간이 걸리는 경로를 역추적하면 S-1-2-1-2-2-1-E 임을 알 수 있다. 

다음은 재귀와 동적프로그래밍을 사용하여 위 과정을 파이썬 코드로 구현해보자. 먼저 재귀를 사용하여 구현한 것이다.

```python
class AssemblyLinesRC:
    """
    초기에 알려진 부분 설정하기
    """
    timeStation = [[7,9,3,4,8,4], [8,5,6,4,5,7]]
    timeBelt = [[2,2,3,1,3,4,3], [4,2,1,2,2,1,2]]
    intCount = 0

   	#함수 만들기
    def Scheduling(self, idxLine, idxStation):
        print("Calculate Scheduling : line, station : ", idxLine, idxStation, "(" , self.intCount, "recursion calls)")

        """
    	탈출 조건문 : 초기 상태를 반환하도록 함
    	"""
        if idxStation == 0:
            if idxLine == 1:
                return self.timeBelt[0][0] + self.timeStation[0][0]
            elif idxLine == 2:
                pass return self.timeBelt[1][0] + self.timeStation[1][0]
		"""
    	재귀 호출을 구성
    	"""
        if idxLine == 1:
            costLine1 = self.Scheduling(1, idxStation-1) + self.timeStation[0][idxStation]
            costLine2 = self.Scheduling(2, idxStation-1) + self.timeStation[0][idxStation] + self.timeBelt[1][idxStation]
        elif idxLine == 2:
            costLine1 = self.Scheduling(1, idxStation-1) + self.timeStation[1][idxStation] + self.timeBelt[0][idxStation]
            costLine2 = self.Scheduling(2, idxStation-1) + self.timeStation[1][idxStation]

        if costLine1 > costLine2:
            return costLine2
        else:
            return costLine1

    """
    주어진 함수를 실행하는 함수 구현
    """
    def startScheduling(self):
        numStation = len(self.timeStation[0])
        costLine1 = self.Scheduling(1, numStation-1) + self.timeBelt[0][numStation]
        costLine2 = self.Scheduling(2, numStation-1) + self.timeBelt[1][numStation]
        if costLine1 > costLine2:
            return costLine2
        else:
            return costLine1
```



다음은 동적 프로그래밍을 사용하여 최단 경로를 구현한 것이다.

```python
class AssemblyLinesDP:
    timeStation = [[7,9,3,4,8,4], [8,5,6,4,5,7]]
    timeBelt = [[2,2,3,1,3,4,3], [4,2,1,2,2,1,2]]

    """
    메모이제이션 테이블 마련하기
    """
    timeScheduling = [list(range(6)), list(range(6))]
    stationTracing = [list(range(6)), list(range(6))]
    
    def startSchedulingDP(self):
        numStation = len(self.timeStation[0])
        self.timeScheduling[0][0] = self.timeStation[0][0] + self.timeBelt[0][0]
        self.timeScheduling[1][0] = self.timeStation[1][0] + self.timeBelt[1][0]
		
        """
    	솔루션 빌딩하기
    	"""
        for itr in range(1, numStation):
            if self.timeScheduling[0][itr-1] > self.timeScheduling[1][itr-1] + self.timeBelt[1][itr]:
                self.timeScheduling[0][itr] = self.timeStation[0][itr] + self.timeScheduling[1][itr-1] + self.timeStation[1][itr]
                self.stationTracing[0][itr] = 1
            else:
                self.timeScheduling[0][itr] = self.timeStation[0][itr] + self.timeScheduling[0][itr-1]
                self.stationTracing[0][itr] = 0

            if self.timeScheduling[1][itr-1] > self.timeScheduling[0][itr-1] + self.timeBelt[0][itr]:
                self.timeScheduling[1][itr] = self.timeStation[1][itr] + self.timeScheduling[0][itr-1] + self.timeStation[0][itr]
                self.stationTracing[1][itr] = 0
            else:
                self.timeScheduling[1][itr] = self.timeStation[1][itr] + self.timeScheduling[1][itr-1]
                self.stationTracing[1][itr] = 1

        costLine1 = self.timeScheduling[0][numStation-1] + self.timeBelt[0][numStation]
        costLine2 = self.timeScheduling[1][numStation-1] + self.timeBelt[1][numStation]

        if costLine1 > costLine2:
            return costLine2, 1
        else:
            return costLine1, 0

        def printTracing(self, lineTracing):
            numStation = len(self.timeStation[0])
            print("Line : ",lineTracing, ", Station : ", numStation)
            for itr in range(numStation-1, 0, -1):
                lineTracing = self.stationTracing[lineTracing][itr]
                print("Line : ",lineTracing, ", Station : ", iter)
```

