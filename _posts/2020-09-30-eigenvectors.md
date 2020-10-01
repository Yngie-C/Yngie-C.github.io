---
layout: post
title: 고윳값(Eigenvalue)과 고유벡터(Eigenvector)
category: Linear Algebra
tag: Linear-Algebra
---



# Eigenvalue & Eigenvector

**고유벡터(Eigenvector)**란 특정 정방행렬 $A$를 통해서 선형 변환했을 때 방향이 변화하지 않는 벡터입니다. 그리고 그 벡터가 변하는 길이는 **고윳값(Eigenvalue)**입니다. 수식으로 나타냈을 때 아래 수식을 만족하는 $\lambda, e$ 를 각각 $A$의 고윳값과 고유벡터라고 합니다.
$$
Ae = \lambda e
$$



$A$는 정방행렬이므로 다음과 같이 나타낼 수 있습니다.



$$
(A - \lambda I)e = 0
$$



$v \neq \vec{0}$ 이면 $\text{det}(A-\lambda I) = 0$ 이어야 합니다. $A - \lambda I$ 의 요소를 나타내면 아래와 같습니다.



$$
A - \lambda I = \left[\begin{array}{cccc} a_{11} - \lambda & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} - \lambda & \cdots & a_{2n} \\ \vdots & \vdots & \ddots \\ a_{n1} & a_{n2} & & a_{nn} - \lambda \end{array} \right]
$$



따라서, 이 행렬의 행렬식을 구하면



$$
\det (A - \lambda I) = \lambda^n + c_1\lambda^{n-1} + c_2\lambda^{n-2} + \cdots
$$



와 같은 $\lambda$ 에 관한 $n$차식이 도출됩니다. 우변의 식인 $\lambda^n + c_1\lambda^{n-1} + c_2\lambda^{n-2} + \cdots + c_n = 0$ 을 만족하는 $\lambda$를 구하여 행렬 $A$의 고윳값을 찾을 수 있습니다. $n \geq 5$ 일 때에는 해를 구하는 공식이 없습니다. 그렇기 때문에 일반적으로 고윳값을 구할 때에는 시행착오법(Trial-error)을 사용하여 구합니다.

$2 \times 2$ 크기의 행렬 $A$를 예로 들어 해당 행렬의 고윳값과 고유벡터를 구해보도록 하겠습니다.



$$
A = \left[\begin{array}{cc} 4 & -5 \\ 2 & -3 \end{array} \right] \\
A - \lambda I = \left[\begin{array}{cc} 4-\lambda & -5 \\ 2 & -3-\lambda \end{array}\right]
$$



$\det (A-\lambda I) = 0$ 인 $\lambda$를 구하면 아래와 같이 나오게 됩니다.


$$
\begin{aligned}
\det (A - \lambda I) &= (4-\lambda)(-3-\lambda)+10 \\
&= \lambda^2-\lambda-12 = (\lambda - 4)(\lambda+3) = 0
\end{aligned} \\
\therefore \lambda  = 4, -3
$$


각 $\lambda$를 $A-\lambda I$ 행렬에 대입하여 나오게 되는 Null space가 각각의 고유벡터가 됩니다. 각 고윳값마다 하나의 고유벡터가 존재합니다.


$$
e = \mathbb{N}(A-\lambda I)
$$


그렇다면 삼각행렬의 고윳값은 어떻게 구할 수 있을까요? 아래와 같은 상삼각행렬 $U$의 고윳값을 구해보겠습니다.


$$
U = \left[\begin{array}{ccc} 1 & 4 & 5 \\ 0 & 5 & 6 \\ 0 & 0 & 3 \end{array} \right]
$$


$U - \lambda I$ 역시 상삼각행렬이고 상삼각 행렬의 행렬식은 피봇의 곱과 같으므로 $\det (U - \lambda I) = 0$ 이 되는 식을 구하면 아래와 같습니다.


$$
\det (U - \lambda I) = (1-\lambda)(5-\lambda)(3-\lambda) = 0 \\
\therefore \lambda = 1,3,5
$$


이렇게 삼각행렬의 고윳값은 해당 행렬의 대각 성분(Diagonal term) 자체가 고윳값이 됩니다.

다음으로 임의의 정방행렬 $A$에 대하여 이 행렬의 대각 성분을 모두 더한 값에 대해 알아보겠습니다. 이 값은 $A$의 Trace(트레이스)라고도 부릅니다. 결과부터 말하면 이 값은 모든 고윳값을 더한 것과 같게 됩니다. 수식으로는 아래와 같이 나타낼 수 있습니다.


$$
\text{Trace of } A = \sum^n_{i=1} a_{ii} = a_{11} + a_{22} + \cdots + a_{nn} = \sum^n_{i=1} \lambda_{ii}
$$


어떻게 이렇게 될 수 있을까요? 임의의 정방행렬 $A$의 고윳값을 구하는 과정을 요소를 사용하여 다시 살펴보겠습니다. $A - \lambda I$ 의 값은 다음과 같이 나타낼 수 있습니다.



$$
A - \lambda I = \left[\begin{array}{ccccc} a_{11} - \lambda & a_{12} & a_{13} & \cdots & a_{1n} \\ a_{21} & a_{22} - \lambda & a_{23} & \cdots & a_{2n} \\  &  & a_{33} - \lambda &  & \\ &  &  & \ddots & \\ a_{n1} & a_{n2} & a_{n3} &  & a_{nn} - \lambda \end{array} \right]
$$


위 행렬의 행렬식을 Co-factor를 사용하여 구한다고 해봅시다. 첫째 행을 기준으로 한다면 수식은 아래와 같을 것입니다.


$$
\det (A - \lambda I) = (a_{11} - \lambda)C_{11} + a_{12}C_{12} + a_{13}C_{13} + \cdots + a_{1n}C_{1n}
$$


Co-factor를 재귀적으로 구하는 과정을 생각해본다면 $C_{11}$ 에는 $\lambda$ 가 $n-1$ 번 등장하지만 다른 Co-factor에는 $\lambda$ 가 $n-2$ 번 등장하는 것을 알 수 있습니다. $1$번째 행에서 한 개의 $\lambda$가 사라지게 되고, $n$번째 열에서 한 개의 $\lambda$가 사라지게 되기 때문입니다. 이를 기억하고 위 수식을 임의의 상수인 계수 $c_1, c_2, \cdots$ 를 사용하여 나타내 보겠습니다.


$$
\det (A - \lambda I) = (-\lambda)^n + c_1(-\lambda)^{n-1} + c_2(-\lambda)^{n-2} + \cdots + c_n
$$


우리의 목표는 위 식이 $0$이 되는 $\lambda$의 값을 구하는 것입니다. 이 때 근과 계수와의 관계를 활용하여 모든 고윳값의 합을 구해볼 수 있습니다. 근과 계수와의 관계에 의하여 $k_1x^n + k_2x^{n-1} + \cdots = 0$ 과 같은 고차방정식에서 모든 근의 합은 $-(k_2/k_1)$ 입니다. 따라서 $\det (A - \lambda I) = 0$ 을 만족하는 모든 고윳값의 합은 $(-\lambda)^{n-1}$의 계수인 $c_1$이 나오게 될 것입니다.

다시 Co-factor로 돌아가 보겠습니다. $C_{11}$에만 $\lambda^{n-1}$ 이 있었고, 나머지 Co-factor에는 최대 차수를 $\lambda^{n-2}$ 으로 갖는 항들을 가지고 있었습니다. 그리하여 최종식에 도출되는 $\lambda^n, \lambda^{n-1}$ 은 오로지 $\det (A - \lambda I)$을 Co-factor로 나타낸 식에서 첫번째 항에 의해서 결정됩니다. 특히 $\lambda^n, \lambda^{n-1}$ 은 아래와 같은 항에 의해 도출되는 것을 알 수 있습니다.


$$
\det (A - \lambda I) = (a_{11} - \lambda)(a_{22} - \lambda) \cdots (a_{nn} - \lambda) + \cdots
$$


이 식으로부터 $(-\lambda)^{n-1}$의 계수인 $c_1$이 어떻게 도출되는 지 구할 수 있습니다.


$$
c_1 = a_{11} + a_{22} + \cdots + a_{nn}
$$


이 값이 행렬 $A$가 가진 모든 고윳값의 합과 같았으므로 $A$의 대각성분을 모두 합한 값, 즉 $A$의 Trace가 모든 고윳값의 합과 같음을 알 수 있습니다.



## Eigenvalue Decomposition

다음은 행렬을 분해하는 또 다른 방법인 **고윳값 분해(Eigenvalue decomposition)**에 대해서 알아보겠습니다. 결과부터 알아보자면 임의의 정방행렬 $A$를 고윳값 분해하여 나타낸 결과는 아래와 같습니다.


$$
A = S \Lambda S^{-1}
$$


위 식에서 행렬 $S$는 각 고유벡터를 열벡터로 갖는 행렬이며, 행렬 $\Lambda$는 고윳값을 대각성분으로 갖는 대각 행렬(Diagonal matrix)입니다. 이렇게 나타낼 수 있는 이유에 대해 알아보겠습니다. 먼저 두 행렬의 곱 $AS$를 아래와 같이 나타낼 수 있습니다.


$$
AS = A \left[\begin{array}{cc} e_1 & e_2 & \cdots & e_n \end{array} \right] = \left[\begin{array}{cc} Ae_1 & Ae_2 & \cdots & Ae_n \end{array} \right]
$$


가장 오른쪽 항의 각 요소를 고윳값과 고유벡터의 정의를 사용하여 아래와 같이 나타낼 수 있습니다.


$$
\left[\begin{array}{cc} Ae_1 & Ae_2 & \cdots & Ae_n \end{array} \right] = \left[\begin{array}{cc} \lambda_1e_1 & \lambda_2e_2 & \cdots & \lambda_ne_n \end{array} \right]
$$


마지막 식은 아래와 같이 나타낼 수 있습니다. 이 과정을 통해 $AS = S \Lambda$임을 알 수 있습니다.


$$
\left[\begin{array}{cc} \lambda_1e_1 & \lambda_2e_2 & \cdots & \lambda_ne_n \end{array} \right] = \left[\begin{array}{cc} e_1 & e_2 & \cdots & e_n \end{array} \right]\left[\begin{array}{cc} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\  \vdots & \vdots & \ddots & \\ 0 & 0 & & \lambda_n \end{array} \right] = S \Lambda \\
\therefore AS = S \Lambda
$$


마지막으로 양 변에 $S^{-1}$을 곱해주어 $A$에 대한 식으로 나타내어 줄 수 있습니다.


$$
A = S \Lambda S^{-1}
$$



## Properties

고윳값과 고유벡터의 정의와 고윳값 분해를 사용하면 고윳값과 고유벡터가 가진 여러가지 성질을 알 수 있습니다. 첫 번째 성질은 각각 다른 값을 갖는 고윳값 $\lambda_1, \lambda_2, \cdots, \lambda_n$ 에 대하여 고유벡터 $e_1, e_2, \cdots, e_n$ 가 선형 독립(Linearly independent)이라는 점입니다. 이는 귀류법을 통해서 증명할 수 있습니다.

먼저 $e_2$ 가 $e_1$ 에 종속이라고 가정해보겠습니다. 그렇다면 $e_2 = c \cdot e_1$ 처럼 한 벡터를 다른 벡터의 스칼라 곱으로 나타낼 수 있을 것입니다. 이 식의 양변에 행렬 $A$를 곱하면 아래와 같이 나타낼 수 있습니다.


$$
Ae_2 = c \cdot Ae_1
$$


고윳값과 고유벡터의 정의에 의하여 위 식의 좌변과 우변은 각각 아래 식처럼 변형해 줄 수 있습니다.


$$
\lambda_2e_2 = c \cdot \lambda_1 e_1
$$


그리고 원래 $e_2 = c \cdot e_1$ 식의 양변에 스칼라 값인 $\lambda_2$를 곱하여 식을 나타내면 $\lambda_2e_2 = c \cdot \lambda_2e_1$ 로 나타낼 수 있습니다. 두 식의 좌변은 $\lambda_2e_2$ 로 동일하므로 한 식의 우변을 대입한 뒤에 모든 항을 좌변으로 넘긴 뒤 공통 부분으로 묶어내면 아래와 같이 나타낼 수 있습니다.


$$
c (\lambda_2 - \lambda_1)e_1 = 0
$$


고유벡터인 $e_1$은 영벡터가 아니고, 두 고윳값이 같지 않으므로 $c = 0$이 됩니다. 즉, $e_2$ 와 $e_1$이 종속 관계에 있다는 가정은 틀리게 되는 것이며 두 벡터가 선형 독립 관계에 있음을 알 수 있습니다. 이를 일반화하여 다음과 같이 나타낼 수 있습니다. 특정한 고유벡터 $e_n$가 다른 고유벡터와 선형 종속 관계에 있다고 하면 아래와 같이 나타낼 수 있습니다.


$$
e_n = c_1e_1 + c_2e_2 + \cdots +c_{n-1}e_{n-1}
$$


양변에 $A$를 곱한 식은 각각 아래와 같이 나오게 됩니다.


$$
Ae_n = c_1Ae_1 + c_2Ae_2 + \cdots +c_{n-1}Ae_{n-1} \\
\lambda_ne_n = c_1\lambda_1e_1 + c_2\lambda_2e_2 + \cdots + c_{n-1}\lambda_{n-1}e_{n-1}
$$


그리고 양변에 $\lambda_n$을 곱한 식은 아래와 같습니다.


$$
\lambda_ne_n = c_1\lambda_ne_1 + c_2\lambda_ne_2 + \cdots + c_{n-1}\lambda_ne_{n-1}
$$


두 수식을 빼주면 식이 다음과 같이 변하게 됩니다.


$$
c_1 (\lambda_n - \lambda_1)e_1 + c_2 (\lambda_n - \lambda_2)e_2 + \cdots + c_{n-1} (\lambda_n - \lambda_{n-1})e_{n-1} = 0
$$


이를 만족하려면 $c_1, c_2, \cdots, c_{n-1} = 0$이어야 하므로 다른 고윳값을 갖는 고유벡터는 나머지 고유벡터와 선형 독립 관계에 있는 것을 알 수 있습니다.

두 번째 성질은 고윳값 분해에 사용되는 $S$가 Unique하지 않다는 점입니다. 특정한 고유벡터 $e$를 스칼라배 해준 $ke$ 역시 고유벡터이기 때문에 만약 특정 행렬의 $S$가 아래 식의 $S_1$과 같다면 $S_2$ 역시 해당 행렬의 $S$가 됩니다.


$$
S_1 = \left[\begin{array}{cc} 1 & 1 \\ 1 & -1 \end{array} \right] \qquad S_2 = \left[\begin{array}{cc} 1 & -1 \\ 1 & 1 \end{array} \right]
$$



다음 성질은 행렬 $A$의 거듭제곱과 고윳값의 관계를 나타낸 것입니다. 특정 행렬 $A$ 의 거듭제곱인 $A^k$ 에 대하여 아래와 같은 식이 성립합니다.


$$
A^ke = \lambda^ke
$$



위 식은 위에서 알아보았던 고윳값 분해를 사용하여 증명할 수 있습니다. 고윳값 분해를 사용하여 행렬의 거듭제곱 $A^k$ 를 다음과 같이 나타낼 수 있습니다.

 

$$
\begin{aligned}
A^k &= (S \Lambda S^{-1})^k \\
&= S \Lambda S^{-1}S \Lambda S^{-1}S \Lambda S^{-1} \cdots S \Lambda S^{-1} \\
&= S \Lambda^k S^{-1} \\
\therefore A^kS &= \Lambda^kS
\end{aligned}
$$

