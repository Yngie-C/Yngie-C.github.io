---
layout: post
title: 파이썬 코딩의 기술 (2)
category: Python
tag: Python
---

 

아래 내용은 [파이썬 코딩의 기술](http://www.yes24.com/Product/Goods/25138160) (브렛 슬라킨, 길벗, 2016)를 참조하여 작성하였습니다.

<br/>

## 14) None을 반환하기보다는 예외를 일으키자

아래의 코드를 보자.

```python
def divide(a,b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

x, y = 0, 5
result = divide(x, y)
if not result:
    print('Invalid inputs')
```

본래 위 코드의 의도는 분모가 0일 경우에 'Invalid inputs'를 출력하는 것이지만, 분자가 0일 경우에도 같은 경고문이 발생한다. 이런 상황은 `None` 에 특별한 의미가 있을 때 파이썬 코드에서 흔히 하는 실수다. 이러한 실수를 피하는 방법은 2가지다.

1. 반환 값을 두 개로 나누어 튜플에 담기 : 튜플의 첫 번째 부분은 작업이 성공했는지 실패했는지를 알려주고, 두 번째 부분은 계산된 실제 결과를 할당한다. 

```python
def divide(a,b):
    try:
        return True, a / b
    except ZeroDivisionError:
        return False, None

success, result = divide(x, y)
if not success:
    print('Invalid inputs')
```

하지만 위 코드에서 사용하지 않을 변수인 `success` 를 관례에 따라 `_` 로 입력할 경우 위에서 발생했던 문제가 똑같이 발생하게 된다.

2. 절대로 None을 반환하지 않기( **더 좋은 방법** ) : 대신 호출하는 쪽에 예외를 일으켜서 호출하는 쪽에서 그 예외를 처리하도록 한다. 아래의 코드를 보자.

```python
def divide(a,b):
    try:
        return a / b
    except ZeroDivisionError as e:
        raise ValueError('Invalid inputs') from e

x, y = 5, 2
try:
    result = divide(x, y)
except ValueError:
    print('Invalid inputs')
```

위 코드에서 `divide` 함수 내에서는 오류를 발생시키고, 호출하는 쪽에서 이 오류를 처리함으로서 `None` 이 유발하는 오류를 처리할 수 있다.

<br/>

## 15) 클로저가 변수 스코프와 상호 작용하는 방법을 알자

숫자 리스트를 정렬할 때, 특정 그룹의 숫자들이 먼저 오도록 우선순위를 매기려고 한다고 하자. 이렇게 만드는 일반적인 방법은 리스트의 sort메소드에 헬퍼 함수를 key인수로 넘기는 것이다. 헬퍼의 반환 값은 리스트에 있는 각 아이템을 정렬하는 값으로 사용된다. 헬퍼는 주어진 아이템이 중요한 그룹에 있는지 확인하고 그에 따라 정렬 키를 다르게 할 수 있다.

```python
def sort_priority(values, group):
    def helper(x):
        if x in group:
            return(0, x)
        return(1, x)
    values.sort(key=helper)
    
numbers = [8, 3, 1, 2, 5, 4, 7, 6]
group = {2, 3, 5, 7}
sort_priority(numbers, group)
print(numbers)
>>> [2, 3, 5, 7, 1, 4, 6, 8]
```

위 결과가 도출되는 이유는 세 가지다.

- 파이썬은 클로저(closure)를 지원한다. 클로저란 자신이 정의된 스코프에 있는 변수를 참조하는 함수다. 바로 이 점 덕분에 helper 함수가 sort_priority의 group 인수에 접근할 수 있다. 

<br/>

## 16) 리스트를 반환하는 대신 제너레이터를 고려하자



<br/>

## 17) 인수를 순회할 때는 방어적으로 하자



<br/>

## 18) 가변 위치 인수로 깔끔하게 보이게 하자



<br/>

## 19) 키워드 인수로 선택적인 동작을 제공하자



<br/>

## 20) 동적 기본 인수를 지정하려면 None과 docstring을 사용하자



<br/>

## 21) 키워드 전용 인수로 명료성을 강요하자



<br/>

