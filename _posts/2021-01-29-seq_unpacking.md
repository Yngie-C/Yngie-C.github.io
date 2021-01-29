---
layout: post
title: 반복형 언패킹 (Iterable Unpacking)
category: Python
tag: Python
---

 

아래 내용은 [전문가를 위한 파이썬 - 루시나누 하말류(한빛미디어, 2016)](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=88728476)을 참조하여 작성하였습니다. 예시 코드는 모두 직접 작성하였습니다. 아래 게시물과 관련된 파이썬 공식 문서는 [PEP 3132 -- Extended Iterable Unpacking](https://www.python.org/dev/peps/pep-3132/) 에서 확인하실 수 있습니다.



# 반복형 언패킹

파이썬에는 리스트, 튜플, 문자열 등 다양한 시퀀스(Seqeunce)가 있습니다. 이번 게시물에서는 시퀀스를 언패킹하는 **반복형 언패킹(Iterable Unpacking)**에 대해서 알아보겠습니다.



## 병렬 할당

**병렬 할당(Pararell assignment)**이란 시퀀스의 요소를 각 변수에 할당하는 것을 말합니다. 병렬 할당으로 리스트를 언패킹하는 아래의 코드를 보겠습니다.

```python
>>> number_list = [1, 2, 5]
>>> a, b, c = number_list
>>> print(f"a : {a}, b : {b}, c : {c}")

a : 1, b : 2, c : 5
```

리스트의 각 값이 `a,b,c` 에 순서대로 할당된 것을 확인할 수 있습니다. 병렬 할당을 확장하면 변수 간 값을 교환하는 코드도 간단하게 나타낼 수 있습니다. 파이썬에서는 다음과 같이 한 줄로 두 변수의 값을 교환할 수 있습니다.

```python
>>> x, y = y, x
```

꼭 두 변수의 값만 교환할 수 있는 것은 아닙니다. 이 방법을 사용해서 언패킹 했던 `a,b,c` 의 값을 바꾸어 보겠습니다.

```python
>>> c, a, b = a, b, c
>>> print(f"a : {a}, b : {b}, c : {c}")

a : 2, b : 5, c : 1
```

세 변수의 값이 할당 해준대로 (`c = a, a = b, b = c`) 잘 바뀐 것을 확인할 수 있습니다.

병렬 할당은 반복문에서도 자주 사용됩니다. 병렬 할당을 사용하여 요소가 튜플 `(idx, prime_num)` 인 `idx` 번째 소수 `prime_num` 을 리스트를 `idx, prime_num`을 각각 `key, value` 로 갖는 딕셔너리로 만들 수 있습니다.

```python
>>> prime_idx_dict = {}
>>>
>>> for idx, prime_num in prime_number_list:
>>>    prime_idx_dict[idx] = prime_num
>>>
>>> prime_idx_dict

{1: 2, 2: 3, 3: 5, 4: 7, 5: 11, 6: 13, 7: 17, 8: 19}
```



## 초과된 항목을 가져오는 *(Asterisk)

함수를 선언할 때 초과되는 파라미터(parameter)를 `*args` 로 나타내듯 파이썬에서는 *(Asterisk)를 붙여 초과된 인수를 가져올 수 있습니다. 이 방법을 사용하면 시퀀스를 좀 더 다양하게 언패킹 할 수 있습니다.

```python
>>> first, second, *others = (1, 2, 3, 4, 5)
>>> print(f"first : {first}, second : {second}, others : {others}")

first : 1, second : 2, others : [3, 4, 5]
```

Asterisk는 시퀀스의 마지막 부분 뿐만 아니라 앞부분과 가운데 부분도 가져올 수 있습니다. 아래의 코드를 보겠습니다.

```python
# 앞부분 가져오기
>>> *others, last = ('a', 'b', 'c', 'd', 'e')
>>> print(f"others : {others}, last : {last}")

others : ['a', 'b', 'c', 'd'], last : e

# 가운데 부분 가져오기
>>> first, second, *others, last = ('u', 'w', 'x', 'y', 'z')
>>> print(f"first : {first}, second : {second}, others : {others}, last : {last}")

first : u, second : w, others : ['x', 'y'], last : z
```

가져와 주기로 지정했던 `first`, `second`, `last` 를 제외한 나머지 요소가 `others` 에 리스트로 할당된 것을 확인할 수 있습니다. 만약 요소가 초과하여 할당될 요소가 없는 경우에는 빈 리스트를 반환합니다.

 ```python
# 예시 1
>>> first, second, *others = (1, 2)
>>> print(f"first : {first}, second : {second}, others : {others}")

first : 1, second : 2, others : []

            
# 예시 2
>>> first, second, *others, last = ('x', 'y', 'z')
>>> print(f"first : {first}, second : {second}, others : {others}, last : {last}")

first : x, second : y, others : [], last : z
 ```

이 방법의 한계는 오직 하나의 변수에만 *를 사용할 수 있다는 점입니다. 아래의 코드를 보겠습니다.

```python
>>> first, *others_first, fifth, sixth, *others_second = (2, 3, 5, 7, 11 ,13 ,17, 19)

'SyntaxError: two starred expressions in assignment'
```

위와 같이 "2개의 asterisk를 할당에 사용하지 말라"는 메시지의 `SyntaxError` 가 발생하는 것을 확인할 수 있습니다.



## 내포된 시퀀스 언패킹하기

파이썬에서는 내포된 시퀀스도 언패킹 할 수 있습니다. 말로만 설명하면 이해하기 어려우니 아래의 예시 코드를 보겠습니다. 내포된 리스트의 요소를 언패킹하는 코드입니다.

```python
>>> language, version, [first_data_type, second_data_type, third_data_type]  = ('python', '3.6,9', ['int', 'float', 'string'])
>>> print(f"first_data_type : {first_data_type}, first_data_type : {second_data_type}, third_data_type : {third_data_type}")

first_data_type : int, first_data_type : float, third_data_type : string
```

내포된 리스트의 요소가 제대로 언패킹 된 것을 볼 수 있습니다. 이 때 `[first_data_type, second_data_type, third_data_type]` 에서 리스트를 나타내는 `[]` 를 제거하면 어떻게 될까요? 아래와 같은 코드를 작성하여 보겠습니다.

```python
>>> language, version, first_data_type, second_data_type, third_data_type  = ('python', '3.6,9', ['int', 'float', 'string'])

'ValueError: not enough values to unpack (expected 5, got 3)'
```

위와 같이 `ValueError`가 뜨는 것을 알 수 있습니다. 

내포된 시퀀스 언패킹 방법을 확장하면 3중으로 내포된 시퀀스도 언패킹 할 수 있습니다. 아래의 코드를 실행해 보겠습니다.

```python
>>> language, version, (first_type, second_type, third_type, (fourth_type, fifth_type)) = ('python', '3.6,9', ('int', 'float', 'string', ('list', 'tuple')))
>>> print(f"first_data_type : {first_data_type}, first_data_type : {second_data_type}, third_data_type : {third_data_type}, fourth_type : {fourth_type}, fifth_type : {fifth_type}")

first_data_type : int, first_data_type : float, third_data_type : string, fourth_type : list, fifth_type : tuple
```

제대로 언패킹되는 것을 확인할 수 있습니다. 











