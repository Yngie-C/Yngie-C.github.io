---
layout: post
title: 클로저 (Closure)
category: Python
tag: Python
---

 

아래 내용은 [파이썬 코딩도장](https://dojang.io/mod/page/view.php?id=2366) , [스쿨오브웹](http://schoolofweb.net/blog/posts/파이썬-클로저-closure/) , [헥사브레인](https://blog.hexabrain.net/347) 을 참조하여 작성하였습니다.



# 클로저

위키피디아에서 언급하고 있는 클로저의 정의는 다음과 같다.

>  컴퓨터 언어에서 클로저(Closure)는 일급 객체 함수(first-class functions)의 개념을 이용하여 스코프(scope)에 묶인 변수를 바인딩 하기 위한 일종의 기술이다. 기능상으로, 클로저는 함수를 저장한 레코드(record)이며, 스코프(scope)의 인수(Factor)들은 클로저가 만들어질 때 정의(define)되며, 스코프 내의 영역이 소멸(remove)되었어도 그에 대한 접근(access)은 독립된 복사본인 클로저를 통해 이루어질 수 있다. - [위키피디아 : 클로저]([https://ko.wikipedia.org/wiki/%ED%81%B4%EB%A1%9C%EC%A0%80_(%EC%BB%B4%ED%93%A8%ED%84%B0_%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D)](https://ko.wikipedia.org/wiki/클로저_(컴퓨터_프로그래밍)))

클로저를 알아보기 전에 먼저 일급 객체란 무엇인지에 대해 알아본 후, 클로저가 어떤 방식으로 사용되는 지 알아보자.



## 일급 객체

파이썬은 **'모든 것이 객체(object)다'** 라는 철학을 가지고 있는 언어이다. 이 객체에는 숫자, 문자열, 리스트, 튜플, 딕셔너리 등이 있다. 파이썬은 함수도 이들과 같은 **일급 객체(First-class citizen)** 로 취급한다. 일급 객체란 다른 객체들에 일반적으로 적용 가능한 연산을 모두 지원하는 객체를 가리키며 일급 객체는 다음과 같은 세 가지 조건을 만족한다. 첫 번째로 변수나 데이터 구조 안에 담을 수 있고, 두 번째로 매개변수로 전달이 가능하며, 마지막으로 리턴값으로 반환할 수 있다. 아래의 코드를 보며 파이썬에서 함수가 일급 객체로서 어떻게 사용될 수 있는지 알아보도록 하자.

```python
def answer():
    print(42)
    
answer()

>>> 42
```

아래와 같이 새로운 함수를 정의할 때 `answer()` 함수를 인자로 집어넣어 사용할 수 있다.

```python
def run_something(func):
    func()
    
run_something(answer)

>>> 42
```

아래는 일급 객체 함수의 성질을 이용하여 사칙 연산 함수를 반복문으로 구현한 것이다.

```python
def add(a,b):
    return a + b
def substract(a,b):
    return a - b
def multiply(a,b):
    return a * b
def divide(a,b):
    return a / b

func_list = [add, substract, multiply, divide]
a = int(input("a의 값을 입력하세요 : "))
b = int(input("b의 값을 입력하세요 : "))
for func in func_list:
    print(func.__name__, ":", func(a,b))

>>> a의 값을 입력하세요 : 3
    b의 값을 입력하세요 : 4
    add : 7
    substract : -1
    multiply : 12
    divide : 0.75
```

위 코드에서 각 사칙연산 함수는 리스트 구조 내에 요소로 담겼으며 이를 `for` 반복문을 통해서 한 번에 출력하는 것을 볼 수 있다. 파이썬에서는 위와 같이 함수가 일급 객체로서 작동하므로 변수나 데이터 구조 안에 담거나, 매개 변수로 전달이 가능하거나 리턴값으로 반환할 수 있다. 이런 특징은 아래에서 클로저 개념을 설명하는데 사용되며, 이외에도 파이썬에서 다양하게 많이 쓰이므로 알아두면 좋다.



## 클로저

이제 일급 객체에 대해 이해했으니 **클로저(Closure)** 에 대해 알아보도록 하자. 클로저가 어떻게 사용되는지 아래의 코드를 보며 알아보자.

```python
def outer_func():
    message = 'Hi'

    def inner_func():
        print(message)

    return inner_func()

outer_func()

>>> Hi
```

위 코드에서 *"Hi"* 가 출력되는 과정은 다음과 같다. `outer_func()` 를 호출하면 첫 번째 줄에서 정의된 함수가 호출된다. 정의된 함수에서 가장 먼저 `message` 변수에 문자열 `"Hi"` 를 할당한다. 그리고 내부에 있는 함수인 `inner_func()` 를 정의하고 `return` 하면서 `inner_func()` 를 호출한다. 호출된 내부 함수에서는 저장된 `message` 변수를 출력하므로 이 값이 최종적으로 출력된다. 이 코드를 변형시키면 어떻게 될까?

```python
def outer_func():
    message = 'Hi'

    def inner_func():
        print(message)

    return inner_func	#위 코드에서 ()를 제거하였다.

outer_func()

>>>
```

이 코드를 실행할 경우 출력값이 없는 것을 볼 수 있다.  `inner_func()` 를 호출하지 않고 `inner_func` 함수 오브젝트만을 리턴하였기 때문이다. 아래의 코드처럼 오브젝트를 다른 변수에 할당한 후 변수를 출력하면 함수 `inner_func` 함수 오브젝트만을 리턴하는 것을 알 수 있다.

```python
def outer_func():
    message = 'Hi'

    def inner_func():
        print(message)

    return inner_func	#위 코드에서 ()를 제거하였다.

my_func = outer_func()

print(my_func)

>>>
<function outer_func.<locals>.inner_func at 0x7f0288608680>
```

이 변수를 이용해서 `inner_func` 를 호출하는 코드를 작성해보자.

```python
def outer_func():
    message = 'Hi'

    def inner_func():
        print message

    return inner_func

my_func = outer_func()

my_func()

>>> Hi
```

`inner_func` 가 할당되어 있는 변수 `my_func` 를 통해서 *"Hi"* 라는 결과물을 출력할 수 있었다. 하지만 이 *"Hi"* 는 `outer_func` 함수 내에서 `message` 변수에 할당된 문자열이었다. 어떻게 `inner_func` 가 자신 밖에 있는 `message` 를 참조할 수 있었을까?

 ```python
def outer_func():
    message = 'Hi'

    def inner_func():
        print(message)

    return inner_func

my_func = outer_func()

print(my_func)
print(dir(my_func))
print(type(my_func.__closure__))
print(my_func.__closure__)
print(my_func.__closure__[0])
print(dir(my_func.__closure__[0]))
print(my_func.__closure__[0].cell_contents)

>>>>
<function outer_func.<locals>.inner_func at 0x7f02886085f0>

['__annotations__', '__call__', '__class__', '__closure__', '__code__', '__defaults__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__get__', '__getattribute__', '__globals__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__kwdefaults__', '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__', '__qualname__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']

<class 'tuple'>

(<cell at 0x7f02885a24d0: str object at 0x7f028860d730>,)

<cell at 0x7f02885a24d0: str object at 0x7f028860d730>

['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'cell_contents']

Hi
 ```

위 코드에서 `dir()` 함수는 인자로 전ㄷ라받은 객체가 어떤 변수와 메서드를 갖고 있는지 나열하는 함수이다. 이 중 클로저라는 메서드를 사용했을 때의 타입과 그 값을 출력한다. 출력값을 통해 클로저 메서드는 튜플이고 그 값은 `(<cell at 0x7f02885a24d0: str object at 0x7f028860d730>,)` 임을 알 수 있다. 이 클로저에 어떤 값이 담겨 있는 지를 알아보기 위해 다시 `cell_contents` 메서드를 적용한 뒤 그 값을 출력하면 *"Hi"* 가 나온다. 이를 통해 외부에서 정의된 매개변수인 *"Hi"* 가 `inner_func()` 에 어떻게 전달되는지를 알 수 있다.

이런 클로저의 성질을 다양하게 이용하면 하나의 함수만으로 다양한 결과물을 출력할 수 있다. 아래의 코드는 클로저를 활용하여 두 가지 결과를 만들어내는 파이썬 코드이다. 

```python
def outer_func(tag):
    tag = tag

    def inner_func(txt):
        text = txt
        print('<{0}>{1}<{0}>'.format(tag, text))

    return inner_func

h1_func = outer_func('h1')
p_func = outer_func('p')

h1_func('h1태그의 안입니다.')
p_func('p태그의 안입니다.')

>>> <h1>h1태그의 안입니다.<h1>
    <p>p태그의 안입니다.<p>
```