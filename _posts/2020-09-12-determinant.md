---
layout: post
title: 행렬식(Determinant)
category: Linear Algebra
tag: Linear-Algebra
---





# Determinant

이번에는 **행렬식(Determinant)**에 대해서 알아보겠습니다. 행렬식이란 무엇일까요? 오늘도 역시 위키피디아로 알아보도록 하겠습니다.

> 선형대수학에서, **행렬식**은 정사각 행렬에 스칼라를 대응시키는 함수의 하나이다. 실수 정사각 행렬의 행렬식의 절댓값은 그 행렬이 나타내는 선형 변환이 초부피를 확대시키는 배수를 나타내며, 행렬식의 부호는 방향 보존 여부를 나타낸다.

즉, 행렬식이란 특정한 정방 행렬마다 주어지는 하나의 스칼라값입니다. 특정 정방행렬 $A$의 행렬식은 일반적으로 $\det A$ 로 나타냅니다.

## Formula

그렇다면 행렬식은 어떻게 구할 수 있을까요? 행렬식을 구하는 데에도 [가우스 소거법(Gaussian elimination)](https://yngie-c.github.io/linear%20algebra/2020/02/18/LA2/)이 사용됩니다. $\det A$ 가 행렬 $A$에서 가우스 소거법을 진행하며 생성되는 피봇의 곱으로 정의되기 때문입니다. 수식으로 나타내면 아래와 같습니다.


$$
\det A = \pm \prod^n_{i=1} \text{pivots}
$$


이 됩니다. $\pm$ 이 붙는 이유는 가우스 소거를 진행하는 과정에서 피봇팅을 하면 행을 교차(Row exchange)하게 되는데 이 때문에 부호 교체가 발생하는 것입니다. 행을 교차하는 것과 행렬식 부호의 관계는 아래에 있는 행렬식의 성질에서 다시 알아보겠습니다.

해당 공식을 통하여 $2 \times 2$행렬의 행렬식이 어떻게 나오게 되는 지를 알아보겠습니다.


$$
\det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right] = \det \left[\begin{array}{cc} a & b \\ 0 & d - b(c/a) \end{array} \right] = a(d - b(c/a)) = ad - bc
$$
 

아래에서 Cofactor를 통한 공식을 하나 더 알아볼 것입니다. 하지만 가우스 소거법을 이용한 방식보다 계산량이 많아 실제로는 가우스 소거를 이용한 방법이 더 자주 사용됩니다.

## Basic Properties

행렬식은 세 가지 기본적인 성질을 가지고 있습니다. 첫 번째 성질은 **단위 행렬(Identify matrix)의 행렬식이 $1$**이라는 점입니다.

### Row Exchange

두 번째 특성은 정방행렬 $A$의 **두 행이나 두 열을 바꿀 때마다(Row exchange) 부호가 바뀐다**는 점입니다. $2 \times 2$ 사이즈의 행렬을 사용하여 성질이 진짜로 그러한 지를 알아보겠습니다. 아래 식에서 행을 한 번 바꾸니 부호가 반대가 되는 것을 알 수 있습니다.



$$
\begin{aligned}
\det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right] &= ad - bc \\
\det \left[\begin{array}{cc} c & d \\ a & b \end{array} \right] &= bc - ad = -(ad - bc) \\ &= - \det A
\end{aligned}
$$

## Linearly Dependent on 1st row

세 번째 성질은 행렬식이 첫 번째 행에 대해서 **선형 종속(Linearly dependent)**이라는 점입니다. 이 성질 때문에 아래와 같은 수식이 성립하게 됩니다.

$$
\det \left[\begin{array}{cc} a + a^\prime & b + b^\prime \\ c & d \end{array} \right] = 
\det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right] + \det \left[\begin{array}{cc} a^\prime & b^\prime \\ c & d \end{array} \right]
$$


등호 왼쪽에 있는 행렬식을 공식을 활용하여 풀어 쓰면 아래와 같습니다.


$$
\begin{aligned}
\det \left[\begin{array}{cc} a + a^\prime & b + b^\prime \\ c & d \end{array} \right] &= (a+a^\prime)d - (b+b^\prime)c \\ &= (ad - bc) + (a^\prime d - b^\prime c) \\
&= \det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right] + \det \left[\begin{array}{cc} a^\prime & b^\prime \\ c & d \end{array} \right]
\end{aligned}
$$

위 식을 통해 행렬식이 첫 번째 행에 대해서 선형 종속 관계에 있음을 알 수 있습니다. 세 번째 성질을 이용하면 다음과 같은 수식들도 모두 만족함을 보일 수 있습니다.

$$
\begin{aligned}
1) &\det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right] = \det \left[\begin{array}{cc} a & 0 \\ c & d \end{array} \right] + \det \left[\begin{array}{cc} 0 & b \\ c & d \end{array} \right] \\
2) &\det \left[\begin{array}{cc} t\cdot a & t\cdot b \\ c & d \end{array} \right] = t\cdot \det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right] \\
3) &\det \left[\begin{array}{cc} t\cdot a & t\cdot b \\ t\cdot c & t\cdot d \end{array} \right] = t^2 \cdot \det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right]
\end{aligned}
$$


위 수식들 중 세 번째 식을 일반화하면 특정 정방행렬 $A$의 모든 요소를 스칼라배 해준 행렬 $t \cdot A$의 행렬식을 다음과 같이 나타낼 수 있습니다.


$$
\det (t\cdot A_{n \times n}) = t^n \cdot \det A
$$



## Additional Properties

### 2개(이상)의 동일한 행을 갖는 행렬 A의 행렬식

위에 등장했던 세 가지 성질을 이용하면 행렬식의 추가적인 몇 가지 성질도 알아낼 수 있습니다. 첫 번째는 행렬 $A$에서 두 개(이상)의 행이 같다면 $\det A = 0$ 이라는 점입니다. 특정 행렬 $A$를 전치한 행렬 $A^T$의 열벡터를 각각 $a_1, a_2, \cdots, a_n$ 이라고 하면 행렬 $A$는 다음과 같이 나타낼 수 있을 것입니다.


$$
A = \left[\begin{array}{c} a_1^T \\ a_2^T \\ \vdots \\ a_n^T \end{array} \right]
$$



이 중에서 $i,j$ 번째 행, 즉 $a_i^T, a_j^T ((i \neq j) \leq n)$가 서로 같다고 해보겠습니다. 그렇다면 원래의 행렬 $A$와 $i,j$ 번째 행을 서로 바꾼 행렬을 $A^\prime$ 이라고 하면 각각을 다음과 같이 나타낼 수 있습니다.


$$
A = \left[\begin{array}{c} a_1^T \\ \vdots \\ a_i^T \\ \vdots \\ a_j^T \\ \vdots  \\ a_n^T \end{array} \right] \qquad
A^\prime = \left[\begin{array}{c} a_1^T \\ \vdots \\ a_j^T \\ \vdots \\ a_i^T \\ \vdots  \\ a_n^T \end{array} \right]
$$



위에서 두 행을 한 번 교차(Row exchange)하면 행렬식의 부호가 바뀌는 성질이 있었습니다. 그렇기 때문에 두 행렬의 행렬식의 절댓값은 동일하지만 부호는 다르게 됩니다. 이를 수식으로 나타내면 아래와 같습니다.


$$
\det A^\prime = - \det A
$$

하지만 $a_i = a_j$ 이므로 실제로 두 행렬의 요소는 모두 같습니다. 그렇기 때문에 아래의 식도 성립해야 합니다.


$$
\det A^\prime = \det A
$$


두 식을 연결하면 $\det A = - \det A$ 가 되므로 이를 만족하는 $\det A = 0$ 밖에 없게 됩니다. 



### Zero Row or Column

특정 행렬 $A$ 내에 모든 성분이 $0$인 행이나 열이 존재하면 $A$의 행렬식 $\det A = 0$ 이 됩니다.



### Raw Operation

또 하나의 추가적인 성질은 Row operation을 해도 행렬식 값이 바뀌지 않는다는 점입니다. 이 성질 덕분에 가우스 소거법을 이용하여 행렬식을 구할 수 있게 됩니다. 이 성질을 일반화하여 수식으로 나타내면 아래와 같습니다.



$$
\begin{aligned}
\det \left[\begin{array}{cc} a - lc & b - ld \\ c & d \end{array} \right] &= \det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right] + \det \left[\begin{array}{cc} -lc & -ld \\ c & d \end{array} \right] \\
&=\det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right]-l \cdot \det \left[\begin{array}{cc} c & d \\ c & d \end{array} \right] \\
&= \det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right]
\end{aligned}
$$



### Determinant of AB & Transpose A

다음 성질은 $\det{AB}$ 와 $\det A^T$에 관련된 것입니다. 첫 번째로 두 행렬 $A,B$를 곱한 행렬 $AB$의 행렬식에 대해 알아보겠습니다. 이 경우에 대해서는 일반적으로 아래의 식이 성립하게 됩니다. 아래 식의 증명은 생략하도록 하겠습니다.


$$
\det{AB} = \det{A} \cdot \det{B}
$$


다음으로 전치 행렬 $A^T$ 의 행렬식 $\det A^T$ 에 대해서 알아보겠습니다. 일반적으로 $\det A^T = \det A$가 성립합니다. $LDU$ 분해법을 사용하면 쉽게 증명할 수 있습니다. 아래는 이 성질을 $A$를 각각 $LDU$ 분해한 뒤에 위에서 알아본 것입니다. 행렬곱의 행렬식을 구하는 방법을 통해서 최종적으로 두 행렬 모두 $\det D$ 임을 알아낼 수 있습니다.



$$
\begin{aligned}
\det A = \det (LDU) &= \det L \cdot \det D \cdot \det U = \det D \\
\det A^T = \det (LDU)^T &= \det U^TD^TL^T =\det U^T \cdot \det D^T \cdot \det L^T = \det D \\
&\therefore \det A = \det A^T
\\
\because D^T = D, &\quad (LDU)^T = U^TD^TL^T, \quad \det U = \det L = 1
\end{aligned}
$$



## Another Formula

위에서 알아본 성질을 사용하면 행렬식을 구하기 위한 새로운 공식을 유도할 수 있습니다. 이 공식을 유도하는 데에는 $\det A$ 가 첫 번째 행에 선형 종속이라는 성질, 두 행을 교환할 경우 행렬식의 부호가 바뀌는 성질, 모든 성분이 $0$인 행 또는 열이 있을 때 $\det A = 0$ 인 성질이 사용됩니다.

이 세 가지 성질을 이용하여 $2 \times 2$ 행렬의 행렬식을 구해보겠습니다.



$$
\begin{aligned}
\det \left[\begin{array}{cc} a & b \\ c & d \end{array} \right] =& \det \left[\begin{array}{cc} a & 0 \\ c & d \end{array} \right] + \det \left[\begin{array}{cc} 0 & b \\ c & d \end{array} \right] \\
=& -\det \left[\begin{array}{cc} c & 0 \\ a & 0 \end{array} \right] -\det \left[\begin{array}{cc} 0 & d \\ a & 0 \end{array} \right] -\det \left[\begin{array}{cc} c & 0 \\ 0 & b \end{array} \right] -\det \left[\begin{array}{cc} 0 & d \\ 0 & b \end{array} \right] \\
=& 0 + \det \left[\begin{array}{cc} a & 0 \\ 0 & d \end{array} \right] - \det \left[\begin{array}{cc} c & 0 \\ 0 & b \end{array} \right] + 0 = ad-bc
\end{aligned}
$$



이를 $3 \times 3$ 행렬에도 적용해 보겠습니다. 첫 번째 행에 선형 종속 관계이므로 첫 번째 행의 각 요소에 대해 식을 나누면 아래와 같이 쓸 수 있습니다.



$$
\det \left[\begin{array}{ccc} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{array} \right] = \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{array} \right] + \det \left[\begin{array}{ccc} 0 & a_{12} & 0 \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{array} \right] + \det \left[\begin{array}{ccc} 0 & 0 & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{array} \right]
$$



위 식의 우변의 세 항 중에서 첫 번째에 해당하는 항을 다시 두 번째 행 요소를 기준으로 풀어쓸 수 있습니다. 풀어쓴 식은 아래와 같습니다.

 
$$
\det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{array} \right] = \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ a_{21} & 0 & 0 \\ a_{31} & a_{32} & a_{33} \end{array} \right] + \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ 0 & a_{22} & 0 \\ a_{31} & a_{32} & a_{33} \end{array} \right] + \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ 0 & 0 & a_{23} \\ a_{31} & a_{32} & a_{33} \end{array} \right]
$$



동일하게 위 식의 우변의 세 항 중에서 첫 번째에 해당하는 항을 다시 세 번째 행 요소를 기준으로 풀어쓸 수 있습니다. 풀어쓴 식은 아래와 같습니다.


$$
\det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ a_{21} & 0 & 0 \\ a_{31} & a_{32} & a_{33} \end{array} \right] = \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ a_{21} & 0 & 0 \\ a_{31} & 0 & 0 \end{array} \right] + \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ a_{21} & 0 & 0 \\ 0 & a_{32} & 0 \end{array} \right] + \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ a_{21} & 0 & 0 \\ 0 & 0 & a_{33} \end{array} \right]
$$



이렇게 모든 항을 풀어쓰게 되면 총 27개의 항이 나오게 될 것입니다. 이 중 모든 열의 요소가 $0$이 아닌 항들만 모아보면 아래와 같이 나오게 될 것입니다.



$$
\begin{aligned}
\det \left[\begin{array}{ccc} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{array} \right] = \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ 0 & a_{22} & 0 \\ 0 & 0 & a_{33} \end{array} \right] + \det \left[\begin{array}{ccc} a_{11} & 0 & 0 \\ 0 & 0 & a_{23} \\ 0 & a_{32} & 0 \end{array} \right] + \det \left[\begin{array}{ccc} 0 & a_{12} & 0 \\ a_{21} & 0 & 0 \\ 0 & 0 & a_{33} \end{array} \right]\\ + \det \left[\begin{array}{ccc} 0 & a_{12} & 0 \\ 0 & 0 & a_{23} \\ a_{31} & 0 & 0 \end{array} \right] + \det \left[\begin{array}{ccc} 0 & 0 & a_{13} \\ a_{21} & 0 & 0 \\ 0 & a_{32} & 0 \end{array} \right] + \det \left[\begin{array}{ccc} 0 & 0 & a_{12} \\ 0 & a_{22} & 0 \\ a_{31} & 0 & 0 \end{array} \right]
\end{aligned}
$$



각각의 값을 구하여 나타내면 아래와 같이 나타나게 될 것입니다.


$$
\det A = a_{11}a_{22}a_{33} + (-1)a_{11}a_{23}a_{32} + \cdots = \sum^3_{\alpha, \beta, \gamma} a_{1\alpha}a_{2\beta}a_{3\gamma}(\det P_{\alpha \beta\gamma})
$$



위 식을 $3 \times 3$이 아닌 $n \times n$ 크기의 행렬로 일반화하면 아래와 같이 나타낼 수 있습니다. 이렇게 유도되는 공식을 **Big formula**라고 합니다.


$$
\det A_{n \times n} = \sum_{\alpha, \beta, \gamma, \cdots, \mu} a_{1\alpha}a_{2\beta}a_{3\gamma} \cdots a_{n\mu}(\det P_{\alpha \beta\gamma\cdots \mu})
$$


다시 $3 \times 3$ 행렬로 돌아가서 $a_{11}$과 관련된 아이들만 모아보겠습니다.

 

$$
a_{11}(a_{22}a_{33}\det P_{1} + a_{23}a_{32}\det P_{2}) = a_{11}C_{11}
$$



이 때 등장하는 두 항을 공통된 부분으로 묶어낸 뒤에 나머지 부분을 **Co-factor**, $C$ 라고 나타내겠습니다. 이 때 등장하는 Cofactor는 해당 행과 열의 요소를 제외한 나머지 **부분 행렬(Sub matrix)의 행렬식**과 같게 됩니다. 위 식과 같에 등장하는 $C_{11}$ 은 아래와 같이 나타나게 되는 것이지요.


$$
C_{11} = \det \left[\begin{array}{cc} a_{22} & a_{23} \\ a_{32} & a_{33} \end{array} \right]
$$


일반적으로 나타내면 아래와 같이 나타낼 수있습니다.


$$
C_{ij} = (-1)^{i+j}\det M_{ij} \qquad M =  \text{minor matrix}
$$


 Co-factor를 사용하면 Big formula의 식을 아래와 같이 조금 더 간단하게 나타낼 수 있습니다. 첫 번째 행을 기준으로 할 경우에는 식이 아래와 같이 됩니다.



$$
\det A = a_{11}C_{11} + a_{12}C_{12} + \cdots + a_{1n}C_{1n} = \sum^n_{j=1} a_{1j}C_{1j}
$$



무조건 첫 번째 행을 기준으로 해야하는 것은 아닙니다. 임의의 행인 $i$번째 행을 기준으로 사용하여도 같은 결과를 도출할 수 있습니다.


$$
\det A = a_{i1}C_{i1} + a_{i2}C_{i2} + \cdots + a_{in}C_{in} = \sum^n_{j=1} a_{ij}C_{ij}
$$


하지만 실제로는 Co-factor를 사용하더라도 재귀적으로 계속 부분 행렬(Sub-matrix)의 행렬식을 구해야 합니다. 그렇기 때문에 일반적으로 행렬식을 구할 때에는 가우스 소거법을 통한 방법을 더 많이 사용하고 있습니다.