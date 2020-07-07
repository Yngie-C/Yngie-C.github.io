---
layout: post
title: 리스트(List)에서의 튜플(Tuple) 혹은 객체(Object)의 정렬
category: Python
tag: Python
---

 

아래 내용은 [파이썬 공식 문서](https://docs.python.org/ko/3/) 와 [TWpower님의 깃허브](https://twpower.github.io/posts) 를 참조하여 작성하였습니다.



# 정렬

파이썬에서 리스트를 정렬하는 방법은 두 가지가 있다. 첫 번째는 리스트 자체를 (제자리에서, in-place) 수정하는 `.sort()` 메서드이다. 두 번째는 새로운 정렬된 리스트를 만드는 `sorted()` 내장 함수이다. 또한 `.sort()` 메서드는 리스트에서만 사용할 수 있는데 비해 `sorted()` 함수는 튜플, 딕셔너리 등 모든 이터러블을 받아들일 수 있다는 차이점이 있다.

본 게시물에서는 이 방법들로 튜플(tuple)이나 객체(object)를 요소로 갖는 리스트를 정렬하는 방법에 알아보고자 한다.



## 튜플을 요소로 갖는 리스트

다음과 같은 리스트가 있다고 해보자.

```python
tuples_in_list = [('a',3,2), ('b',7,1), ('b',2,3), ('c',2,9), ('a',12,19), ('c',1,1)]
```

이 함수를 특별한 조건없이 정렬하여 출력하면 결과값은 어떻게 될까?

```python
tuples_in_list = [('a',3,2), ('b',7,1), ('b',2,3), ('c',2,9), ('a',12,19), ('c',1,1)]
tuples_in_list.sort()
print(tuples_in_list)

>>> [('a',3,2), ('a',12,19), ('b',2,3), ('b',7,1), ('c',1,1), ('c',2,9)]
```

위와 같은 결과가 나오는 이유는 파이썬에서는 튜플을 비교할 때 **인덱스를 순차적으로 나아가며 각 인덱스의 값을 비교**하기 때문이다. 먼저 인덱스 0 을 기준으로 모든 튜플을 비교하여 정렬한다. 이 과정이 끝나면 그 안에서 인덱스 1 을 기준으로 비교하여 정렬한다. 앞선 예시에서는 첫 번째로 각 튜플의 0 번째 리스트인 `'a', 'b', 'c'` 를 비교하여 각 튜플을 배치한다. 두 번째로  `('a',3,2), ('a',12,19)` 의 1 번째 리스트인 `3` 과 `12` 를 비교하여 `3` 이 더 작으므로 순서를 바꾸지 않고 그대로 두게 된다. 이런 방식을 `('b',7,1), ('b',2,3)` 와 `('c',2,9), ('c',1,1)` 에 대해서도 진행하면 위와 같은 결과가 나오게 된다.

그렇다면 특정 인덱스에 우선순위를 두고 비교하고 싶을 때는 어떻게 해야 할까? `.sort()`혹은 `sorted()` 내에 있는 `key=` 라는 파라미터(parameter)에 적절한 함수를 지정해주면 된다. 다음은 위 예시와 같은 리스트를 각 튜플의 마지막 인덱스(인덱스 2)에 우선순위를 두고 정렬하는 방법이다.

```python
tuples_in_list = [('a',3,2), ('b',7,1), ('b',2,3), ('c',2,9), ('a',12,19), ('c',1,1)]
sortedlist_by_idx2 = sorted(tuples_in_list, key=lambda tpls:tpls[2])
print(sortedlist_by_idx2)

>>> [('b',7,1), ('c',1,1), ('a',3,2), ('b',2,3), ('c',2,9), ('a',12,19)]
```

위에서 출력된 결과를 보면 일단 튜플의 마지막 인덱스에 속하는 값 `1, 1, 2, 3, 9 ,19` 을 기준으로 정렬되어 있음을 알 수 있다. 마지막 인덱스의 값이 중복되는 `('b',7,1), ('c',1,1)` 의 경우에는 정렬 안정성 덕분에 원래의 순서를 유지하고 있음을 알 수 있다. 정렬 안정성에 대해서는 아래에서 더 자세하게 보도록 한다.



## 객체를 요소로 갖는 리스트

이 방법은 객체를 요소로 갖는 리스트에 대해서도 유용하게 쓸 수 있다. 다음과 같은 클래스와 리스트가 있다고 해보자.

```python
class Student:
    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age
    def __repr__(self):
        return repr((self.name, self.grade, self.age))

student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 17),
    Student('dave', 'B', 10),
]
```

위의 `student_objects` 를 특정 기준에 따라 정렬하고 싶을 때는 튜플 요소를 정렬할 때와 비슷한 방법을 사용하면 된다. 아래는 각각 학생을 이름순, 나이순으로 정렬한 것이다. `key=` 파라미터에 조건에 맞게 잘 정렬되었음을 볼 수 있다.  

```python
stu_obj_by_name = sorted(student_objects, key=lambda student:student.name)
stu_obj_by_age = sorted(student_objects, key=lambda student:student.age)
print(stu_obj_by_name)
print(stu_obj_by_age)

>>> [('dave', 'B', 10), ('jane', 'B', 17), ('john', 'A', 15)]
>>> [('dave', 'B', 10), ('john', 'A', 15), ('jane', 'B', 17)]
```



## operator 모듈 함수

위와 같은 방법은 굉장히 많이 쓰이므로 파이썬에서는 이를 더욱 쉽고 빠르게 해주는 내장 모듈을 제공하고 있다. 파이썬의 `operator` 모듈 내에 있는 `itemgetter(), attrgetter()` 함수는 `key=` 파라미터에 함수를 만들어 지정해주지 않아도 쉽게 이와 같은 결과를 얻을 수 있도록 해준다.

먼저 `itemgetter()` 를 쓰는 예시를 보자. `itemgetter()` 함수는 파라미터로 인덱스를 받으므로 객체를 요소로 갖는 리스트에서는 쓸 수 없다. 인덱스 2를 기준으로 정렬한 결과 위에서 `lambda tpls:tpls[2]` 와 같은 람다(lambda) 함수를 만들어 지정해준 것과 같은 결과를 얻은 것을 볼 수 있다.

```python
from operator import itemgetter

tuples_in_list = [('a',3,2), ('b',7,1), ('b',2,3), ('c',2,9), ('a',12,19), ('c',1,1)]
sortedlist_by_idx2 = sorted(tuples_in_list, key=itemgetter(2))
print(sortedlist_by_idx2)

>>> [('b',7,1), ('c',1,1), ('a',3,2), ('b',2,3), ('c',2,9), ('a',12,19)]
```

다음으로 `attrgetter()` 를 쓰는 예시를 보자. `attrgetter()` 함수는 파라미터로 속성(attribute)을 받으므로 객체를 요소로 갖는 리스트에서만 쓸 수 있다. 나이를 기준으로 정렬한 결과 위에서 `lambda student:student.age` 와 같은 람다(lambda) 함수를 만들어 지정해준 것과 같은 결과를 얻은 것을 볼 수 있다.

```python
from operator import attrgetter

student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 17),
    Student('dave', 'B', 10),
]
stu_obj_by_age = sorted(student_objects, key=attrgetter('age'))
print(stu_obj_by_age)

>>> [('dave', 'B', 10), ('john', 'A', 15), ('jane', 'B', 17)]
```

`itemgetter(), attrgetter()` 를 함수를 사용하면 2개 이상의 기준으로 정렬하는 것도 쉽게 할 수 있다. 아래는 `tuples_in_list ` 를 인덱스 0을 기준으로 먼저 정렬하고 인덱스 2를 기준으로 정렬하는 코드이다. 출력된 결과에서 `('c',1,1), ('c',2,9)` 를 보면 두 가지 우선순위를 기준으로 정렬이 잘 되었음을 알 수 있다. ( `itemgetter(0)` 로만 정렬했다면 정렬 안정성 때문에 원래의 순서를 유지하여 `('c',2,9), ('c',1,1) ` 와 같이 출력된다.)

```python
from operator import itemgetter

tuples_in_list = [('a',3,2), ('b',7,1), ('b',2,3), ('c',2,9), ('a',12,19), ('c',1,1)]
sortedlist_by_idx02 = sorted(tuples_in_list, key=itemgetter(0,2))
print(sortedlist_by_idx02)

>>> [('a',3,2), ('a',12,19), ('b',7,1), ('b',2,3), ('c',1,1), ('c',2,9)]
```

아래는 `student_objects` 에 몇 개의 객체를 더 추가한 `student_object2` 를 이름순으로 먼저 정렬한 뒤 나이순으로 정렬하는 코드이다. 출력된 결과에서 `('john', 'C', 14), ('john', 'A', 15)` 를 보면 두 가지 우선순위를 기준으로 정렬이 잘 되었음을 알 수 있다.

```python
from operator import attrgetter

student_objects2 = [
    Student('john', 'A', 15),
    Student('jane', 'B', 17),
    Student('dave', 'B', 10),
    Student('john', 'C', 14),
    Student('kevin', 'A', 16)
]
stu_obj2_by_name_age = sorted(student_objects, key=attrgetter('name','age'))
print(stu_obj2_by_name_age)

>>> [('dave', 'B', 10), ('jane', 'B', 17), ('john', 'C', 14), ('john', 'A', 15), ('kevin', 'A', 16)]
```



## 정렬 안정성

정렬 안정성(Stability of Sort)이란 같은 키를 가질 때, 원래의 순서가 유지되는 것을 말한다. 예를 들어 `tuples_in_list ` 을 첫 번째 인덱스를 기준으로 정렬한다고 해보자.

```python
from operator import itemgetter

tuples_in_list = [('a',3,2), ('b',7,1), ('b',2,3), ('c',2,9), ('a',12,19), ('c',1,1)]
sortedlist_by_idx0 = sorted(tuples_in_list, key=itemgetter(0))
print(sortedlist_by_idx0)

>>> [('a', 3, 2), ('a', 12, 19), ('b', 7, 1), ('b', 2, 3), ('c', 2, 9), ('c', 1, 1)]
```

파이썬 `.sort()` 메서드나 `sorted()` 내장함수는 [팀 소트(Timsort)](https://orchistro.tistory.com/175) 라는 정렬 알고리즘을 사용한다. 팀 소트는 안정성을 보장하는 정렬 알고리즘이다. 그래서 이를 활용하여 다양한 정렬을 시도할 수 있다. 위에서 등장한 `student_object2` 를 성적에 대해서는 내림차순으로 정렬한 뒤 나이에 대해서는 오름차순으로 정리하는 예시를 보자.

```python
from operator import attrgetter

student_objects2 = [
    Student('john', 'A', 15),
    Student('jane', 'B', 17),
    Student('dave', 'B', 10),
    Student('john', 'C', 14),
    Student('kevin', 'A', 16)
]

stu_obj2_by_age = sorted(student_objects, key=attrgetter('age'))
stu_obj2_by_age_graderev = sorted(stu_obj2_by_age, key=attrgetter('grade'), reverse=True)
print(stu_obj2_by_age_graderev)

>>> [('john', 'C', 14), ('dave', 'B', 10), ('jane', 'B', 17), ('john', 'A', 15), ('kevin', 'A', 16)]
```

이를 다중 패스로 정렬하기 위하여 필드와 순서의 튜플 리스트를 받는 래퍼(wrapper) 함수로 나타낼 수 있다. 추상화한 함수 `multisort()` 를 호출한 뒤 동일한 조건을 적용했을 때 같은 결과가 나오는 것을 볼 수 있다.

```python
def multisort(xs, specs):
    for key, reverse in reversed(specs):
        xs.sort(key=attrgetter(key), reverse=reverse)
    return xs

multisort(list(student_objects2), (('grade', True), ('age', False)))

>>> [('john', 'C', 14), ('dave', 'B', 10), ('jane', 'B', 17), ('john', 'A', 15), ('kevin', 'A', 16)]
```

