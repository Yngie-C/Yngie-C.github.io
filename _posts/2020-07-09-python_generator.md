---
layout: post
title: 제너레이터 (Generator)
category: Python
tag: Python
---

 

아래 내용은 [스쿨오브웹-제너레이터]([http://schoolofweb.net/blog/posts/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%A0%9C%EB%84%88%EB%A0%88%EC%9D%B4%ED%84%B0-generator/](http://schoolofweb.net/blog/posts/파이썬-제너레이터-generator/)) 와 [처음 시작하는 파이썬](http://www.yes24.com/Product/Goods/29289816) , [파이썬 코딩의 기술](http://www.yes24.com/Product/Goods/26850854?scode=032&OzSrank=1) 참조하여 작성하였습니다.



# 제너레이터

**제너레이터(Generator)** 는 파이썬의 시퀀스를 생성하는 객체(object)다. 제너레이터의 장점은 전체 시퀀스를 한 번에 처리하지 않고 하나하나씩 처리한다는데 있다. 그렇기 때문에 시퀀스의 크기가 매우 커진다고 해도 메모리 부담 없이 작업을 수행할 수 있다.



## 제너레이터 함수

일반적으로 함수에서 일련의 결과를 생성할 때 선택하는 가장 간단한 방법은 리스트를 반환하는 것이다. 정수 요소로 이루어진 리스트를 입력받아 각 요소를 세제곱한 값으로 이루어진 리스트를 반환하는 함수가 있다고 하자. 해당 함수를 파이썬 코드로 구현하면 아래와 같이 쓸 수 있다.

```python
def cubic_num(num_lst):
    result = []
    for i in num_lst:
        result.append(i ** 3)
    return result

my_nums = [1,2,3,4,5]
my_cubic_num = cubic_num(my_nums)
print(my_cubic_num)

>>> [1, 8, 27, 64, 125]
```

위와 같이 일련의 시퀀스를 다루는 작업은 제너레이터를 사용해서 구현할 수도 있다. 위 코드를 제너레이터 함수로 구현하면 다음과 같다. 제너레이터 함수는 `return` 대신에 `yield` 표현식을 사용한다.

```python
def cubic_num_gen(num_lst):
    for i in num_lst:
        yield i ** 3

my_nums = [1,2,3,4,5]
my_cubic_num_gen = cubic_num_gen(my_nums)
print(my_cubic_num_gen)

>>> <generator object cubic_num_gen at 0x7f0c50cf2f50>
```

동일한 작업을 수행하는 함수임에도 이상한 결과가 출력되는 것을 알 수 있다. 제너레이터 함수는 호출되면 실제로 실행하지 않고 바로 이터레이터(iterator)를 반환한다. 이터레이터는 내장 함수 `next()` 를 호출할 때 다음 값을 생성해내는 상태를 가진 헬퍼 객체이다. 내장 함수 `next()` 는 제너레이터가 다음 `yield` 표현식으로 진행할 수 있도록 한다. 그리고 제너레이터가 `yield` 에 전달한 값을 이터레이터가 호출하는 쪽에 반환한다.

```python
def cubic_num_gen(num_lst):
    for i in num_lst:
        yield i ** 3

my_nums = [1,2,3,4,5]
my_cubic_num_gen = cubic_num_gen(my_nums)

print(next(my_cubic_num_gen))

>>> 1
```

한 번에 하나의 작업만을 수행하는 제너레이터 특성상 첫 번째 요소에 대해서만 진행된 것을 볼 수 있다.  `next()` 함수를 반복적으로 사용하여 값을 출력해보자. 입력 리스트의 크기가 5이므로 5번 사용하면 모든 값을 출력할 수 있다.

```python
def cubic_num_gen(num_lst):
    for i in num_lst:
        yield i ** 3

my_nums = [1,2,3,4,5]
my_cubic_num_gen = cubic_num_gen(my_nums)

print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))

>>> 1
    8
    27
    64
    125
```

요소의 개수보다 더 많은  `next()` 함수를 호출할 경우에는 더 이상 진행할 작업이 없으므로 `StopIteration` 예외가 발생한다.

```python
def cubic_num_gen(num_lst):
    for i in num_lst:
        yield i ** 3

my_nums = [1,2,3,4,5]
my_cubic_num_gen = cubic_num_gen(my_nums)

print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))
print(next(my_cubic_num_gen))

>>> 1
    8
    27
    64
    125
----------------------------------------------------------------
"""StopIteration:"""
```

매번 `next()` 함수를 통해서 이터레이터를 진행시킬 수는 없다. 그래서 일반적으로는 `for` 반복문을 통해 제너레이터 함수를 호출하여 사용한다. 다음의 예시를 보자.

```python
def cubic_num_gen(num_lst):
    for i in num_lst:
        yield i ** 3

my_nums = [1,2,3,4,5]
my_cubic_num_gen = cubic_num_gen(my_nums)

for j in my_cubic_num_gen:
    print(j)
    
>>> 1
    8
    27
    64
    125
```

 `for` 반복문은 `StopIteration` 예외가 발생할 때까지 이터레이터를 진행시키기 때문에 시퀀스 내에 존재하는 모든 요소에 대해 해당 작업을 순차적으로 수행한다.



## 리스트와 제너레이터로 큰 시퀀스 처리하기

글의 서두에서 언급한 것처럼 제너레이터의 장점은 큰 시퀀스를 처리하는 데에 있다. 한 번에 하나의 요소만을 처리하기 때문에 시퀀스의 크기가 커지더라도 메모리를 많이 사용하지 않는다. 아래 코드는 요소개 100만 개인 시퀀스를 리스트와 제너레이터로 처리할 때 걸리는 시간과 메모리 사용량을 나타낸 것이다.

```python
from __future__ import division
import os
import psutil술
import random as r
import time

languages = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java', 'Bash/Shell/Powershell', 'C#', 'PHP', 'TypeScript','C++','C']
frameworks = ['jQuery', 'React.js', 'Angular', 'ASP.NET', 'Express', 'ASP.NET Core', 'Vue.js', 'Spring', 'Angular.js', 'Django', 'Flask']

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss/1024/1024

def lang_framework_list(num_people):
    result = []
    for i in range(num_people):
        person = {
            'id': i,
            'language': r.choice(languages),
            'framework': r.choice(frameworks) 
        }
        result.append(person)
    return result
        
t1 = time.process_time()
lang_framework = lang_framework_list(1000000)
t2 = time.process_time()
mem_after = process.memory_info().rss/1024/1024
total_time = t2 - t1

print('시작 전 메모리 사용량: {} MB'.format(mem_before))
print('종료 후 메모리 사용량: {} MB'.format(mem_after))
print('총 소요된 시간: {:.6f} 초'.format(total_time))

>>> 시작 전 메모리 사용량: 47.90625 MB
    종료 후 메모리 사용량: 335.4765625 MB
    총 소요된 시간: 1.323855 초
```

100만개의 요소로 이루어진 시퀀스를 한 번에 처리하려다 보니 메모리 사용량이 크게 늘어난 것을 볼 수 있으며 작업을 수행하는 데 걸린 시간도 1초가 넘는 것을 볼 수 있다.

```python
from __future__ import division
import os
import psutil
import random as r
import time

languages = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java', 'Bash/Shell/Powershell', 'C#', 'PHP', 'TypeScript','C++','C']
frameworks = ['jQuery', 'React.js', 'Angular', 'ASP.NET', 'Express', 'ASP.NET Core', 'Vue.js', 'Spring', 'Angular.js', 'Django', 'Flask']

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss/1024/1024

def lang_framework_generator(num_people):
    for i in range(num_people):
        person = {
            'id': i,
            'language': r.choice(languages),
            'framework': r.choice(frameworks) 
        }
        yield person
        
t1 = time.process_time()
lang_framework = lang_framework_generator(1000000)
t2 = time.process_time()
mem_after = process.memory_info().rss/1024/1024
total_time = t2 - t1

print('시작 전 메모리 사용량: {} MB'.format(mem_before))
print('종료 후 메모리 사용량: {} MB'.format(mem_after))
print('총 소요된 시간: {:.6f} 초'.format(total_time))

>>> 시작 전 메모리 사용량: 48.20703125 MB
    종료 후 메모리 사용량: 48.20703125 MB
    총 소요된 시간: 0.000065 초
```

제너레이터로 작업을 진행할 경우 시퀀스가 커져도 전후 메모리 사용량에 변화가 없으며 소요된 시간도 리스트를 한꺼번에 처리하는 것보다 훨씬 더 짧아지는 것을 볼 수 있다.



## 제너레이터 표현식(Generator Expression)

파이썬에는 리스트로부터 리스트를 빠르게 생성하는 리스트 컴프리헨션(List Comprehension)이 있다. 컴프리헨션은 세트(Set)와 딕셔너리(Dictionary)에도 적용할 수 있다. 이런 컴프리헨션을 잘 사용하면 알고리즘을 작성할 때 파생되는 자료구조를 간명하게 생성할 수 있다. 하지만 위에서 리스트로 작업을 수행했던 것과 같이 큰 시퀀스에 대해 리스트 컴프리헨션을 사용하면 메모리 사용량이 커지고 시간이 오래 걸리게 된다.

파이썬에서는 이런 문제를 해결하기 위해 제너레이터에도 컴프리헨션과 비슷한 표현을 적용할 수 있도록 해놓았다. 이를 **제너레이터 표현식(Generator Expression)** 이라고 하며 대괄호 `[ ]` 대신 소괄호 `( )` 를 사용하여 나타낸다. 아래는 1000만 개의 요소를 세제곱하는 작업을 각각 리스트 컴프리헨션과 제너레이터 표현식을 사용하여 수행한 결과이다. 위에서 한 것과 유사한 결과가 도출되는 것을 볼 수 있다.

```python
from __future__ import division
import os
import psutil
import time

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss/1024/1024

def cubic_num_comp(num):
    return [i**3 for i in range(num)]

t1 = time.process_time()
my_cubic_num = cubic_num_comp(10000000)
t2 = time.process_time()
mem_after = process.memory_info().rss/1024/1024
total_time = t2 - t1

print('시작 전 메모리 사용량: {} MB'.format(mem_before))
print('종료 후 메모리 사용량: {} MB'.format(mem_after))
print('총 소요된 시간: {:.6f} 초'.format(total_time))

>>> 시작 전 메모리 사용량: 48.046875 MB
    종료 후 메모리 사용량: 589.55078125 MB
    총 소요된 시간: 2.111658 초
```

리스트 컴프리헨션을 사용한 경우에는 메모리 사용량이 높은 것을 볼 수 있다.

```python
from __future__ import division
import os
import psutil
import time

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss/1024/1024

def cubic_num_comp(num):
    yield (i**3 for i in range(num))

t1 = time.process_time()
my_cubic_num = cubic_num_comp(10000000)
t2 = time.process_time()
mem_after = process.memory_info().rss/1024/1024
total_time = t2 - t1

print('시작 전 메모리 사용량: {} MB'.format(mem_before))
print('종료 후 메모리 사용량: {} MB'.format(mem_after))
print('총 소요된 시간: {:.6f} 초'.format(total_time))

>>> 시작 전 메모리 사용량: 48.0859375 MB
    종료 후 메모리 사용량: 48.0859375 MB
    총 소요된 시간: 0.000049 초
```

제너레이터 표현식을 사용하면 시간도 매우 짧으며 메모리 사용량의 변화도 없는 것을 볼 수 있다.















