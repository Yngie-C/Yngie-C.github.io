---
layout: post
title: Collections Module
category: Python
tag: Python
---

 

아래 내용은 [파이썬 자료구조와 알고리즘](http://www.yes24.com/Product/Goods/74971408) , [파이썬 공식문서](https://docs.python.org/3/library/collections.html) 을 참조하여 작성하였습니다.



# collections

**collections** 은 리스트(list), 튜플(tuple), 딕셔너리(dictionary) 등에 대한 확장 데이터 구조를 제공하는 파이썬 내장 모듈이다. collections 내에는 



## Counter

**Counter** (카운터)는 해시 가능한 객체를 카운팅하는 데에 특화된 서브클래스이다. 시퀀스 내에 있는 요소의 개수를 딕셔너리 형태로 반환한다.

```python
from collections import Counter

seq1 = [1, 2, 3, 2, 1, 1, 2, 2, 4, 3]
Counter(seq1)

>>> Counter({1: 3, 2: 4, 3: 2, 4: 1})
```

Counter 내에도 다양한 메서드가 존재한다. 그 중 하나는 `.most_common()` 이다. 이 메서드는 자연수 값 N을 인자로 받는다. 이 메서드는 큰 value를 가지는 요소와 그 개수를 상위 N개 만큼 출력한다. 이를 활용하면 어떤 요소가 해당 자료형 내에 많은 지를 알 수 있다. 해당 문자열 혹은 시퀀스에 들어있는 요소의 종류 수와 같거나 그보다 클 경우에는 모든 요소에 대해 출력한다.

```python
from collections import Counter

print(Counter('bibbidibobbidiboo').most_common(3))
print(Counter('bibbidibobbidiboo').most_common(7))

>>> [('b', 7), ('i', 5), ('o', 3)]
    [('b', 7), ('i', 5), ('o', 3), ('d', 2)]
```

`.subtract()`  는 해당 요소 개수의 차이를 구할 때 사용되는 메서드이다. 아래의 코드를 보며 이해해보자.

```python
from collections import Counter

counter_bibbidi = Counter('bibbidibobbidiboo')
counter_banana = Counter('bananaboat')

print(counter_bibbidi)
print(counter_banana)

counter_bibbidi.subtract(counter_banana)

print(counter_bibbidi)

>>> Counter({'b': 7, 'i': 5, 'o': 3, 'd': 2})
    Counter({'a': 4, 'b': 2, 'n': 2, 'o': 1, 't': 1})
    Counter({'b': 5, 'i': 5, 'd': 2, 'o': 2, 't': -1, 'n': -2, 'a': -4})
```

`'bibbidibobbidiboo'` 에 포함된 요소의 개수를 카운터를 통해 나타내면 `Counter({'b': 7, 'i': 5, 'o': 3, 'd': 2})` 이다. `'bananaboat'` 에 포함된 요소의 개수를 카운터를 통해 나타내면 `Counter({'a': 4, 'b': 2, 'n': 2, 'o': 1, 't': 1})` 이다. `.subtract()` 메서드는 한 카운터에서 다른 카운터를 빼는 연산을 수행한다. 전자에서 후자를 빼준 후 다시 원래의 카운터를 출력하면 연산이 수행된 이후의 결과가 나오는 것을 볼 수 있다.



## deque

**deque** (데크)는 'Double Ended QUEue' 의 줄임말로서 양쪽 끝에서 모두 큐(Queue)의 역할을 수행할 수 있는 자료구조이다. 일반적인 큐는 LIFO(Last In First Out) 이지만 deque는 말 그대로 양쪽 끝에서 추가와 삭제를 모두 수행할 수 있다. 파이썬이 기본적으로 제공하는 리스트는 리스트의 끝에서만 추가 `.append()` 와 삭제 `.pop()` 및 확장 `extend()` 을 할 수 있다. 하지만 deque에서는 `.appendleft()` 와 `.popleft()` , `extendleft()` 를 통해서 왼쪽에서도 작업을 수행할 수 있다. 

```python
from collections import deque

deq = deque(['c', 'd', 'e'])
deq.append('f')
deq.appendleft('b')
print(deq)

deq.pop()
print(deq)

deq.popleft()
print(deq)

>>> deque(['b', 'c', 'd', 'e', 'f'])
    deque(['b', 'c', 'd', 'e'])
    deque(['c', 'd', 'e'])
```

위 코드에서 `.appendleft()` 혹은 `.popleft()` 를 통해 왼쪽에서도 추가, 삭제 등의 작업을 수행하는 것을 볼 수 있다. 

deque에서는 앞/뒤로 추가, 삭제가 자유롭기 때문에 요소를 순환시킬 수 있는 메서드인 `.rotate()` 도 제공하고 있다.

```python
from collections import deque

deq = deque(['a', 'b', 'c', 'd', 'e'])
deq.rotate(3)
print(deq)

>>> deque(['c', 'd', 'e', 'a', 'b'])
```

리스트에서 사용되었던 메서드인 `.remove()` , `reverse()` , `index()` , `insert()` 등은 그대로 사용할 수 있다. 





## defaultdict

**defaultdict** 는 일반 딕셔너리와 달리 지정되지 않은 키의 값(value)을 자동으로 할당한다. 자동으로 할당되는 값의 자료형은 defaultdict를 선언하면서 지정해줄 수 있다. 아래의 코드를 보고 defaultdict가 동작하는 방식을 알 수 있다.

```python
from collections import defaultdict

dDict_int = defaultdict(int)
print(dDict_int['key1'])

dDict_int['key2'] = 'value2'
print(dDict_int)

>>> 0
    defaultdict(<class 'int'>, {'key1': 0, 'key2': 'value2'})
```

defaultdict를 선언하면서 기본 자료형을 `int` 로 설정해주면 값을 지정해주지 않은 키에 대해 기본 값인 0을 알아서 할당하는 것을 볼 수 있다. `int` 가 아닌 다른 자료형을 사용할 수도 있다. 아래는 기본 자료형을 `list` 로 했을 때의 코드이다.

```python
from collections import defaultdict

dataset = {(1,'a'), (2,'b'), (3,'c'), (2,'d'), (1,'e'), (3,'f')}
dDict_list = defaultdict(list)

for key, value in dataset:
    dDict_list[key].append(value)
    
print(dDict_list)

>>> defaultdict(<class 'list'>, {3: ['c', 'f'], 2: ['d', 'b'], 1: ['a', 'e']})
```

미리 각 키 `1, 2, 3` 에 해당하는 기본 값 `[]` 이 있으므로 `.append()` 만을 통해서 값(value)을 변경할 수 있다.



## namedtuple

`namedtuple` 은 일반 튜플과 성능이 비슷하지만 튜플 항목(field)에 대해 이름이 지어져 있어서 인덱스가 아닌 이름으로도 접근할 수 있는 시퀀스 데이터 타입이다.

```python
from collections import namedtuple

Student = namedtuple('Student', 'name score grade')
student1 = Student('Jackson', 91, 'A')

print(student1[1])
print(student1.score)
print(student1.grade)

>>> 91
    91
    'A'
```

일반적으로 튜플은 값을 변경할 수 없는 불변 자료형이다. 하지만 `namedtuple` 의 경우에는 `._replace` 메서드를 활용하여 특정한 이름의 값을 변경하여 새로운 변수에 저장하거나 기존 변수에 저장하여 덮어씌울 수 있다.

```python
from collections import namedtuple

Student = namedtuple('Student', 'name score grade')
student1 = Student('Jackson', 91, 'A')
student2 = student1._replace(score=95, grade='A+')
print(student2)
print(student2.grade)

>>> Student(name='Jackson', score=95, grade='A+')
    'A+'
```



## OrderedDict

**Orderdict** (정렬 딕셔너리)는 요소가 입력되는 순서를 기억하는 딕셔너리이다. 

```python
from collections import OrderedDict

dataset = {(1,'a'), (2,'b'), (3,'c')}
ODict1 = OrderedDict(dataset)

for key in ODict:
    print(key, ODict[key])
    
>>> 1 'a'
    3 'c'
    2 'b'
```

키가 입력된 순서를 기억하여 추후에 값이 바뀌더라도 순서가 바뀌지 않는 것을 아래 코드에서 볼 수 있다.

```python
ODict[1] = 'f'
print(ODict)

>>> OrderedDict([(1, 'f'), (3, 'c'), (2, 'b')])
```

파이썬 3.7 이후부터는 표준 딕셔너리도 입력된 순서를 보장한다. 그래서 해당 버전 이상의 파이썬을 사용하고 있다면 굳이 OrderedDict를 사용하지 않아도 된다.

```python
dict1 = {1:'a', 2:'b', 3:'c'}
dict1[1] = 'f'
print(dict1)

>>> {1: 'f', 2: 'b', 3: 'c'}
```