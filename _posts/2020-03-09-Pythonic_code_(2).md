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

## 2) PEP 8 스타일 가이드를 따르자

문법만 잘 지킨다면 코드를 마음대로 작성해도 괜찮지만, 일관성 있는 스타일을 사용하면 유지보수가 훨씬 쉬워지고 가독성도 높아진다. **PEP 8** (Python Enhencement Proposal #8)은 파이썬 코드를 어떻게 구성할지 알려주는 스타일 가이드다. 전체적인 내용은 [공식 가이드](https://www.python.org/dev/peps/pep-0008/) 를 참조하는 것이 좋으며 아래는 그 중 중요한 몇 가지를 나열하였다.

- **Whitespace** : 파이썬에서 공백은 문법적으로 의미가 있다. 특히 코드의 명료성 때문에 공백에 민감한 편이다.
  - Tap이 아닌 Space로 들여쓴다. (대개의 Editor에서는 Tap을 4 * Space로 자동으로 변환해준다.)
  - 문법적으로 의미있는 들여쓰기는 각 수준마다 스페이스 4개를 사용한다.
  - 한 줄의 문자 길이가 79자 이하여야 한다.
  - 표현식이 길어서 다음 줄로 이어지면 일반적인 들여쓰기 수준에 추가로 스페이스 4개를 사용한다.
  - 파일에서 함수와 클래스는 빈 줄 2개로 구분해야 한다.
  - 클래스에서 메서드는 빈 줄 하나로 구분해야 한다.
  - 리스트 인덱스, 함수 호출, 키워드 인수 할당에는 스페이스를 사용하지 않는다.
  - 변수 할당 앞뒤에는 스페이스를 하나만 사용한다.
- **Naming** : PEP 8은 언어의 부분별로 독자적인 명명 스타일을 제안한다.
  - 함수, 변수, 속성은 `lowercase_underscore` 형식을 따른다.
  - 보호(protected) 인스턴스 속성은 `_leading_underscore` 형식을 따른다.
  - 비공개(private) 인스턴스 속성은 `__double_leading_underscore` 형식을 따른다.
  - 클래스와 예외는 `CapitalizedWord` 형식을 따른다.
  - 모듈 수준 상수는 `ALL_CAPS` 형식을 따른다.
  - 클래스의 인스턴스 메서드에서는 첫 번째 파라미터(해당 객체를 참조)의 이름을 self로 지정한다.
  - 클래스 메서드에서는 첫 번째 파라미터(해당 클래스를 참조)의 이름을 cls로 지정한다.
- **Expression & Method**
  - 긍정 표현식의 부정( `if not a is b` ) 대신에 인라인 부정( `if a is not b` )을 사용한다.
  - 길이를 확인( `if len(somelist) == 0` )하여 빈 값([] 또는 ' ')을 확인하지 않는다. 대신, `if not somelist` 를 사용하고, 빈 값은 암시적으로 `False` 가 된다고 가정한다.
  - 비어있지 않은 값( `[1]` 또는 `'hi'` ) 에도 위와 같은 방식이 적용된다. 값이 비어있지 않으면 `if somelist` 문이 암시적으로 `True` 가 된다.
  - 한 줄로 된 if문, for과 while 루프, except 복합문을 쓰지 않는다. 이런 문장은 여러 줄로 나누어 명료하게 작성한다.
  - 항상 파일의 맨 위에 import 문을 놓는다.
  - 모듈을 import 할 때는 항상 모듈의 절대 이름을 사용하며 현재 모듈의 경로를 기준으로 상대 경로로 된 이름을 사용하지 않는다. 예를 들어 bar 패키지의 foo 모듈을 임포트하려면 그냥 `import foo` 가 아닌 `from bar import foo` 라고 해야 한다.
  - 상대적인 import 를 해야 한다면 명시적인 구문을 써서 `from . import foo` 라고 한다.
  - import 는 '표준 라이브러리 모듈, 서드파티 모듈, 자신이 만든 모듈' 섹션 순으로 구분해야 한다. 각각의 하위 섹션에서는 알파벳 순서로 import 한다.

<br/>

## 3) bytes, str, unicode의 차이점을 알자

파이썬 3에서는 `bytes` 와 `str` 두 가지 타입으로 문자 시퀀스를 나타낸다. `bytes` 인스턴스는 Raw 8비트 값을 저장한다. `str` 인스턴스는 유니코드 문자를 저장한다. (파이썬 2의 관한 내용은 생략)

유니코드 문자를 바이너리 데이터로 표현하는 방법은 많다. 가장 일반적인 인코딩은 *UTF-8* 이다. 유니코드 문자를 바이너리 데이터로 나타내려면 encode 메서드를 사용해야 하며, 반대로 바이너리 데이터를 유니코드 문자로 나타내려면 decode 메서드를 사용해야 한다. 외부에 제공할 인터페이스에는 유니코드를 인코드하고 디코드해야 한다.

문자 타입이 분리되어 있는 탓에 파이썬 코드에서는 일반적으로 다음 두 가지 상황에 부딪힌다.

- *UTF-8* (혹은 다른 인코딩)으로 인코드된 문자인 Raw 8비트 값을 처리하려는 상황
- 인코딩이 없는 유니코드 문자를 처리하려는 상황

 이 두 경우 사이에서 변환하고 코드에서 원하는 타입과 입력값의 타입이 정확히 일치하게 하려면 헬퍼함수 2개가 필요하다.

```python
def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
	return value	# str 인스턴스

def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode('utf-8')
    else:
        value = bytes_or_str
	return value	# bytes 인스턴스
```

파이썬에서 Raw 8비트 값과 유니코드 문자를 처리할 때 중요한 것이 있다. (파이썬 2의 경우 제외)

- 파이썬 3에서 내장 함수 `open` 이 반환하는 파일 핸들을 사용하는 연산은 기본적으로 UTF-8 인코딩을 사용한다는 점이다. 예를 들어, 파이썬 3에서 임의의 바이너리 데이터를 파일에 기록하려 한다면 다음과 같은 코드를 사용해야 한다. (파이썬 3에서 바이너리 함수를 열 때는 모드에 'r', 'w' 대신 'rb', 'wb'를 써준다.)

```python
with open('/tmp/random.bin', 'wb') as f:
    f.write(os.urandom(10))
```

<br/>

## 4) 복잡한 표현식 대신 헬퍼 함수를 작성하자

파이썬의 간결한  문법을 사용하면 많은 로직을 한 줄로 작성할 수 있다. 하지만 너무 많이 줄여 표현한다면 읽기 어렵다는 단점이 새로 발생하게 된다. 때문에 특정 로직을 반복해서 사용해야 할 때는 헬퍼 함수를 만들어 사용하는 것이 좋다. 표현식이 복잡해지기 시작하면 최대한 빨리 해당 표현식을 작은 조각으로 분할하고 로직을 헬퍼 함수로 옮기는 방안을 고려해야 한다. 무조건 짧은 코드를 만들기보다는 가독성을 선택하는 것이 낫다. 

<br/>

## 5) 시퀀스를 슬라이스 하는 방법을 알자

파이썬은 시퀀스를 **슬라이스** (Slice)해서 조각으로 만드는 문법을 제공한다. 슬라이싱 문법의 기본 형태는 `someList[start:end]` 이며, 여기서 `start` 인덱스는 포함되고, `end` 인덱스는 제외된다. 리스트의 처음부터 슬라이스 할 때는 start의 0을 생략하며, 끝까지 슬라이스 할 때도 end의 마지막 인덱스를 생략한다.

```python
a[:5]	# a[0:5] 에서 첫 번째 인덱스인 0은 쓰지 않는다.
a[5:]	# a[5:len(a)] 에서 마지막 인덱스인 len(a)는 쓰지 않는다.
```

리스트의 끝을 기준으로 할 때는 음수로 슬라이스 하는 것이 편하다.

```python
a[:20]	# 리스트의 처음부터 20개의 원소 추출
a[-20:]	# 리스트의 마지막까지 20개의 원소 추출
```

슬라이싱의 결과는 완전히 새로운 리스트다. 물론 원본 리스트에 있는 객체에 대한 참조는 유지된다. 하지만 슬라이스한 결과를 수정해도 원본 리스트에 아무런 영향을 미치지 않는다.

```python
a = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
b = a[4:]
print(b)
>>> ['d', 'e', 'f', 'g']
b[1] = 99
print(b)
>>> ['d', '99', 'f', 'g']
print(a)
>>> ['a', 'b', 'c', 'd', 'e', 'f', 'g']
```

할당에 사용하면 슬라이스는 원본 리스트에서 지정한 범위를 대체한다. 튜플 할당과 달리 슬라이스 할당의 길이는 달라도 된다. 할당 받은 슬라이스 앞뒤 값은 유지된다. 리스트는 새로 들어온 값에 맟춰 늘어나거나 줄어든다.

```python
a[2:6] = [99, 22, 14]
print(a)
>>> ['a', 'b', 99, 22, 14, 'g']
```

시작과 끝 인덱스를 모두 생략하고 슬라이스 하면 원본 리스트의 복사본을 얻는다. 슬라이스에 시작과 끝 인덱스를 지정하지 않고 할당하면 슬라이스의 전체 내용을 참조 대상의 복사본으로 대체한다.

<br/>

## 6) 한 슬라이스에 start, end, stride를 함께 쓰지 말자.

파이썬에는 위에서 살펴본 기본 슬라이싱뿐만 아니라 `somelist[start:end:stride]` 처럼 슬라이스의 간격을 설정하는 특별한 문법도 있다. 이 문법을 사용하면 시퀀스를 슬라이스할 때 매 n번째 아이템을 가져올 수 있다.

문제는 `stride` 문법이 종종 예상치 못한 동작을 해서 버그를 만들어 내기도 한다는 점이다. 예를 들어, 바이트 문자열을 역순으로 만들기 위해 `stride` -1로 문자열을 슬라이스 한다고 하자. 

```python
x = b'mongoose'	#b는 바이트 문자열로 지정해주기 위함
y = x[::-1]
print(y)
>>> b'esoognom'
```

위 코드는 바이트 문자열이나 ASCII 문자에는 잘 작동하지만, UTF-8 바이트 문자열로 인코딩된 유니코드 문자에는 원하는대로 작동하지 않는다. 

```python
w = '謝謝'
x = w.encode('utf-8')
y = x[::-1]
z = y.decode('utf-8')
>>> UnicodeDecodeError: 'utf-8' codec ~
```

또한 `start` 와 `end` 에 따라 `stride` 부분이 매우 혼란스러울 수 있다. 이런 문제를 방지하려면 `stride` 를 `start` , `end` 인덱스와 함께 사용하지 말아야 한다. `stride` 를 사용해야 한다면 양수 값을 사용하고, `start` 와 `end` 인덱스는 생략하는 것이 좋다. 꼭 셋을 함께 사용해야 한다면, `stride` 를 적용한 결과를 변수에 할당하고, 이 변수를 슬라이스한 결과를 다른 변수에 할당해서 사용하자.

 ```python
a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
b = a[::2]	# ['a', 'c', 'e', 'g']
c = b[1:-1]	# ['c', 'e']
 ```

슬라이싱부터 하고 스트라이딩을 하면 데이터의 얇은 복사본이 추가로 생긴다. 때문에, 첫 번째 연산은 결과로 나오는 슬라이스의 크기를 최대한 줄여야 한다. 프로그램에서 두 과정에 필요한 시간과 메모리가 충분하지 않다면 내장모듈 `itertools` 의 `islice` 메서드를 사용해보자. `islice` 메서드는 `start` , `end` , `stride` 에 음수 값을 허용하지 않는다.

<br/>

## 7) map과 filter 대신 List Comprehension을 사용하자

인수가 하나뿐인 함수를 적용하는 상황이 아니면, 간단한 연산에는 리스트 컴프리헨션이 내장 함수 `map` 보다 명확하다. `map` 을 쓰려면 아래와 같이 계산에 필요한 `lambda` 함수를 생성해야 하는데 별로 깔끔해보이지 않는다.

```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squares = map(lambda x: x**2, a)
```

`map` 과 달리 리스트 컴프리헨션을 사용하면 리스트에 있는 아이템을 간편하게 걸러내서 그에 대응하는 출력을 결과에서 삭제할 수 있다. 리스트 내부의 숫자들 중 짝수만 제곱하는 컴프리헨션은 다음과 같다.

```python
even_squares = [x**2 for x in a if x%2 == 0]
>>> [4, 16, 36, 64, 100]
```

아래와 같이 내장함수 filter를 map과 연계해서 사용해도 같은 결과를 얻을 수 있지만 훨씬 읽기 어렵다.

```python
alt = map(lambda x:x**2, filter(lambda x: x%2 == 0))
```

딕셔너리와 세트에도 컴프리헨션에 해당하는 문법이 있다.컴프리헨션 문법을 쓰면 알고리즘을 작성할 때 파생되는 자료 구조를 쉽게 생성할 수 있다.

<br/>

## 8) List Comprehension에서  표현식을 두 개 넘게 쓰지 말자

리스트 컴프리헨션은 다중 루프도 지원한다. 2중 리스트(행렬)를 모든 셀이 포함된 평평한 리스트로 간략화하는 컴프리헨션 코드는 다음과 같다. 컴프리헨션 내에서 for 문을 2개 사용한다.

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix for x in row]
>>> [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

다중 루프의 또 다른 사용법은 리스트의 레이아웃을 중복해서 구성하는 것이다. 2차원 행렬을 변형하지 않고 각 값을 제곱하는 컴프리헨션은 다음과 같다.

```python
squared = [[x**2 for x in row] for row in matrix]
>>> [[1, 4, 9], [16, 25, 36], [49, 64, 81]]
```

위의 표현식들을 (더 많은 for 가 필요한) 다른 루프에 넣는다면 컴프리헨션이라도 여러 줄로 구분해야 할 만큼 길어진다.

```python
my_lists = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], ...]
    ...
]
# 이면 이 3중 리스트를 평평한 리스트로 만드는 컴프리헨션 코드는 다음과 같다.
flat = [x for sublist1 in my_lists
       for sublist2 in sublist1
       for x in sublist2]
```

이렇게 되면 컴프리헨션이 다른 방법보다 많이 짧지 않다. 오히려 일반 루프문으로 아래와 같이 표현했을 때 들여쓰기 덕분에 더 이해하기가 좋다.

```python
flat=[]
for sublist1 in my_lists:
    for sublist2 in sublist1:
        flat.extend(sublist2)
```

컴프리헨션도 다중 if 조건을 지원한다. 같은 루프 레벨에 여러 조건이 있으면 암시적인 and 표현식이 있다. 아래는 4보다 큰 짝수 값을 가지고 오는 컴프리헨션 들이다.

```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [x for x in a if x>4 if x%2 == 0]
c = [x for x in a if x>4 and if x%2 == 0] # b == c
```

조건은 루프의 각 레벨에서 for 표현식 뒤에 설정할 수 있다. 다음은 행렬에서 행의 합이 10 이상이고 그 행렬의 값 중에 3으로 나누어 떨어지는 것을 제곱하여 반환하는 함수다.

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
filtered = [[x for x in row if x%3 == 0]
           for row in matrix if sum(row) >= 10]
print(filtered)
>>> [[6], [9]]
```

위와 같은 코드는 다른 사람들이 이해하기 어렵기 때문에 피하는 것이 좋다. 대개 리스트 컴프리헨션을 사용할 때는 표현식이 두 개를 넘어가면 파하는게 좋다. 조건 2개, 루프 2개, 혹은 조건 1개와 루프 1개 정도면 된다. 이것보다 복잡해지면 if 문과 for 문을 사용하고 헬퍼 함수를 작성해야 한다.

<br/>

## 9) Comprehension이 클 때는 제너레이터 표현식을 고려하자

리스트 컴프리헨션의 문제점은 입력 시퀀스에 있는 각 값별로 아이템을 하나씩 담은 새 리스트를 통째로 생성한다는 점이다. 입력이 클 때는 메모리를 많이 소모하기 때문에 프로그램을 망가뜨리는 원인이 되기도 한다. 다음은 파일 내에 있는 텍스트 각 줄의 길이를 반환하는 컴프리헨션이다.

```python
value = [len(x) for x in open('/tmp/my_file.txt')]
print(value)
>>> [100, 57, 15, 1 ...]
```

위와 같은 코드를 작성할 때 컴프리헨션으로 하면 파일에 있는 각 줄의 길이만큼 메모리가 필요하다. 즉, 파일 안에 많은 양의 텍스트가 있다면 문제가 발생할 수도 있다. 파이썬은 이런 문제를 해결하기 위해 컴프리헨션과 제너레이터를 일반화한 **제너레이터 표현식** (Generator Expression)을 제공한다. 제너레이터 표현식은 실행될 때 출력 시퀀스를 모두 구체화(메모리에 로딩)하지 않는대신 표현식에서 한 번에 한 아이템을 내어주는 **이터레이터** (Iterator)로 평가된다.

제너레이터 표현식은 () 문자 사이에 리스트 컴프리헨션과 비슷한 문법을 사용하여 생성한다. 다음은 이전 코드와 동일한 기능을 제너레이터 표현식으로 표현한 것이다. 하지만 제너레이터 표현식은 즉시 이터레이터로 평가되므로 더 이상 진행되지는 않는다.

```python
it = (len(x) for x in open('/tmp/my_file.txt'))
print(it)
>>> <generator object <genexpr> at 0x101b8148>
```

제너레이터 표현식에서 다음 출력을 생성하려면 내장함수 `next` 를 사용하여 반환받은 이터레이터를 하나씩 전진시키면 된다.

```python
print(next(it))
print(next(it))
>>> 100
57
```

제너레이터 표현식의 또 하나의 장점은 다른 제너레이터 표현식과 함께 사용할 수 있다는 점이다. 다음은 위 제너레이터 표현식이 반환한 이터레이터를 다른 제너레이터 표현식의 입력으로 사용한 예시다. 

```python
roots = ((x, x**0.5) for x in it)
```

이 이터레이터를 전진시킬 때마다 내부 이터레이터를 전진 시킬뿐만 아니라 조건 표현식을 계산하여 입력과 출력을 처리하게 된다.

```python
print(next(roots))
>>> (15, 3.8729833...)
```

이처럼 제너레이터를 연결하면 파이썬에서 매우 빠르게 실행할 수 있다. 입력 스트림이 큰 경우 그것에 동작하는 기능을 결합하는 방법 중 제너레이터 표현식이 최선의 도구다. 주의할 점은 제너레이터 표현식이 반환한 이터레이터에는 상태가 있으니 이터레이터를 한 번 넘게 사용하지 않도록 주의해야 한다는 것이다.

<br/>

## 10) range 보다는 enumerate를 사용하자

내장 함수인 `range`는 정수 집합을 순회(iterate)하는 루프를 실행할 때 유용하다. 종종 리스트를 순회하거나 리스트에서 현재 아이템의 인덱스를 알고 싶은 경우가 있다. 다음의 코드는 좋아하는 아이스크림의 순위를 출력하는 코드다.

```python
for i in range(len(flavor_list)):
    flavor = flavor_list[i]
    print('%d : %s' % (i + 1, flavor))
```

위 코드는 리스트의 길이를 알아내야 하고, 배열을 인덱스로 접근하며, 읽기 불편하다는 단점이 있다. 파이썬은 이런 상황을 위한 내장 함수 `enumerate` 를 제공한다. `enumerate` 는 지연 제너레이터(Lazy generator)로 이터레이터를 감싼다. 이 제너레이터는 이터레이터에서 루프 인덱스와 다음 값을 한 쌍으로 가져온다. 위의 코드를 `enumerate` 를 활용해 작성한 코드는 다음과 같다.

```python
for i, flavor in enumerate(flavor_list):
    print('%d : %s' % (i + 1, flavor))
```

`enumerate` 로 세기 시작할 숫자를 지정하면 코드를 더욱 짧게 만들 수 있다. 

```python
for i, flavor in enumerate(flavor_list, 1):
    print('%d : %s' % (i + 1, flavor))
```

<br/>

## 11) 이터레이터를 병렬로 처리하려면 zip을 사용하자 

리스트 컴프리헨션을 사용하면 소스 리스트(Source list)에 표현식을 적용하여 파생 리스트(Derived list)를 쉽게 얻을 수 있다. 다음의 코드 같은 경우가 그것이다.

```python
names = ['Cecilia', 'Lise', 'Marie']
letters = [len(n) for n in names]
```

파생 리스트의 아이템과 소스 리스트의 아이템은 서로의 인덱스로 연관되어 있음. 따라서 두 리스트를 병렬로 순회하려면 소스 리스트인 names의 길이만큼 순회하면 된다. 아래의 코드를 보자

```python
longest_name = None
max_letters = 0

for i in range(len(names)):
    count = letters[i]
    if count > max_letters:
        longest_name = names[i]
        max_letters = count

print(longest_name)
>>> 'Cecilia'
```

코드는 잘 돌아가지만, 전체 루프문의 가독성이 좋지는 않다. names와 letters를 인덱스로 접근하면 코드를 읽기 어려워진다. 10) 에 있는 `enumerate` 를 사용하여 코드를 나타내면 조금 더 개선되긴 하지만 그것도 만족스럽지는 못하다.

```python
for i, name enumerate(names):
    count = letters[i]
    if count > max_letters:
        longest_name = names[i]
        max_letters = count
```

파이썬은 위와 같은 상황에서 쓸 수 있는  `zip` 이라는 내장 함수를 지원한다. 파이썬 3에서 `zip` 은 지연 제너레이터로 두 개 이상의 이터레이터를 감싼다. `zip`  제너레이터는 각 이터레이터로부터 다음 값을 담은 튜플을 얻어온다. `zip` 제너레이터를 사용한 코드는 다중 리스트에서 인덱스로 접근하는 코드보다 명료하다.

```python
for name, count in zip(names, letters):
    if count > max_letters:
        longest_name = name
        max_letters = count
```

`zip` 을 사용할 때의 문제점은 입력 이터레이터들의 길이가 다르면 zip이 이상하게 동작한다는 점이다. 이외에도 많은 경우에 `zip` 의 잘라내기 동작은 이상하고 나쁘다. `zip` 으로 실행할 리스트의 길이가 같다고 확신할 수 없다면 대신 내장 모듈 *itertools* 의 `zip_longest` 를 사용하는 방안을 고려해보자.

<br/>

## 12) for과 while 루프 뒤에는 else 블록을 쓰지 말자

파이썬에는 루프에서 반복되는 내부 블록 바로 다음에 `else` 블록을 둘 수 있다.  아래의 예시를 보자.

```python
for i in range(3):
    print('Loop %d' % i)
else:
    print('Else block!')
>>>
Loop 0
Loop 1
Loop 2
Else block!
```

위의 코드를 실행하면 루프문이 끝나자마자 `else` 문이 실행된다. 파이썬에서 `else, except, finally` 개념을 처음 접하는 사람들은 `for/else` 의 `else` 부분이 '루프가 완료되지 않는다면 이 블록을 실행한다'는 의미라고 짐작할 수도 있다. 이런 오해 때문에 루프 다음에는 `else` 블록을 두는 대신 각각의 헬퍼 함수를 두는 것이 좋다. 



<br/>

## 13) try/except/else/finally 에서 각 블록의 장점을 이용하자

파이썬에서는 예외 처리 과정에서 동작을 넣을 수 있는 4번의 구분되는 시점이 있다. `try/except/else/finally` 블록 기능으로 각 시점을 처리한다. 각 블록은 복합문에서 독자적인 목적이 있으며, 이 블록들을 다양하게 조합하면 유용하다.

- `finally`

예외를 전달하고 싶지만 예외가 발생하더라도 정리 코드를 실행하고 싶을 때 `try/finally` 를 사용하면 된다. 일반적인 사용 예 중 하나는 파일 핸들러를 제대로 종료하는 작업이다.

```python
handle = open('/tmp/random_data.txt')	#IOError가 발생할 수 있음
try:
    data = handle.read()	#UnicodeDecodeError가 발생할 수 있음
finally:
    handle.close()	# try: 이후에 항상 실행됨
```

`read` 메서드에서 발생한 예외는 항상 호출 코드까지 전달되며, `close` 메서드 또한 `finally` 블록에서 실행되는 것이 보장된다. 

- `else`

코드에서 어떤 예외를 처리하고, 어떤 예외를 전달할지를 명확히 하려면 `try/except/else` 를 사용하자. 예를 들어 문자열에서 json 딕셔너리 데이터를 로드하여 그 안에 든 키 값을 반환하는 코드가 있다고 하자.

```python
def load_json_key(data, key):
    try:
        result_dict = json.loads(data)	#ValueError가 발생할 수 있음
    except ValueError as e:
        raise KeyError from e
    else:
        return result_dict[key]	#KeyError가 발생할 수 있음
```

- 모두 함께 사용하기

복합문 하나로 모든 것을 처리하고 싶다면 `try/except/else/finally` 를 사용하면 된다. 여기서 `try` 블록은 파일을 읽고 처리하는데 사용하고, `except` 블록은 `try` 블록에서 일어난 예외를 처리하는 데 사용한다. `else` 블록은 파일을 즉석에서 업데이터하고 이와 관련된 예외가 전달되게 하는 데 사용한다. `finally` 블록은 파일 핸들을 정리하는 데 사용한다.

```python
UNDEFIND = object()

def divide_json(path):
    handle = open(path, 'r+')	#IOError가 발생할 수 있음
    try:
        data = handle.read()	#UnicodeDecodeError가 발생할 수 있음
        op = json.loads(data)	#ValueError가 발생할 수 있음
        value = (
            op['numerator'] /
            op['denominator']	#ZeroDivisionError가 발생할 수 있음
        )
    except ZeroDivisionError as e:
        raise UNDEFIND
    else:
        op['result'] = value
        result = json.dumps(op)
        handle.seek(0)
        handle.write(result)	#IOError가 발생할 수 있음
        return value
    finally:
        handle.close()	#항상 실행함
```

<br/>

