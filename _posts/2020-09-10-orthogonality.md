---
layout: post
title: 선형 독립 (Linearly Independence)과 직교성(Orthogonality)
category: Linear Algebra
tag: Linear-Algebra
---



# Orthogonality

이번 게시물에서는 벡터의 **직교성(Orthogonality)**에 대해 알아보겠습니다. 벡터의 직교성은 왜 중요할까요? 첫 번째는 직교하는(Orthogonal) 벡터는 서로 **선형 독립(Linearly Independence)** 관계에 있기 때문입니다. 선형 독립인 벡터 $\vec{v_1}, \vec{v_2}, \cdots, \vec{v_n}$ 는 다음의 조건을 만족합니다.


$$
\begin{aligned}
&c_1\vec{v_1} + c_2\vec{v_2} + \cdots + c_n\vec{v_n} = 0 \\
\text{only,} \quad &c_1 = c_2 = \cdots = c_n = 0
\end{aligned}
$$


두 번째는 직교하는 벡터를 서로 내적하면 결과값이 $0$이 되기 때문입니다. 이런 특성 덕분에 특정 벡터를 직교 벡터의 합으로 나타내면 연산이 매우 쉬워지게 됩니다. 특정 벡터의 크기를 구하는 경우가 이에 속합니다. $\vec{x} = (1,2,3)$의 크기 $\Vert \vec{x} \Vert$ 를 아래와 같이 구할 수 있습니다. 먼저 직교하는 벡터 $\vec{a},\vec{b},\vec{c}$로 나눕니다.


$$
\text{if} \quad \vec{a} = (1,0,0), \vec{b} = (0,2,0),\vec{c} = (0,0,3) \\
\vec{x} = \vec{a}+\vec{b}+\vec{c}
$$


이를 사용하면 아래와 같은 방법을 사용하여 $\Vert \vec{x} \Vert$를 구할 수 있습니다.


$$
\begin{aligned}
\Vert \vec{x} \Vert^2 = \vec{x}\cdot \vec{x} &= (\vec{a}+\vec{b}+\vec{c})\cdot (\vec{a}+\vec{b}+\vec{c}) \\ 
&=\vec{a}\cdot \vec{a}+\vec{b}\cdot \vec{b}+\vec{c} \cdot \vec{c} \\
&= \Vert \vec{a} \Vert^2 + \Vert \vec{b} \Vert^2 + \Vert \vec{c} \Vert^2 \\
&= 1 + 4 + 9 = 13 \\
\therefore \Vert \vec{x} \Vert &= \sqrt{13} \\
\because \vec{a} \cdot \vec{b} &= 0, \vec{b} \cdot \vec{c}=0, \vec{c} \cdot \vec{a} = 0 \quad \text{(Orthogonal)}
\end{aligned}
$$


## Orthogonal Subspace

**직교하는 부분 공간(Orthogonal subspace)**이란 한 부분 공간 내의 모든 벡터가 다른 한 부분 공간 내의 모든 벡터에 대해 직교 관계에 있을 때의 부분 공간 사이의 관계를 말합니다.

대표적인 직교 부분 공간의 예시는 주요 부분 공간(Fundamental subspace)에서 찾을 수 있습니다. 간단한 예시를 통해서 다음 행렬 $A$ 의 주요 부분 공간이 어떻게 그려지는지 알아보겠습니다.


$$
A = \left[\begin{array}{cc} 1 & 2 \\ 3 & 6 \end{array} \right] \xrightarrow{\text{G.E.}} \left[\begin{array}{cc} 1 & 2 \\ 0 & 0 \end{array} \right]
$$


위 식을 사용하여 $A$의 Column space와 Null space를 구할 수 있습니다.


$$
\mathbf{C}(A) = c\left[\begin{array}{c} 1 \\ 3 \end{array} \right] \qquad \mathbf{N}(A) = x_2\left[\begin{array}{c} -2 \\ 1 \end{array} \right]
$$


이번에는 $A^T$를 사용하여 $A$의 Row space와 Left-Null space를 구할 수 있습니다.


$$
A^T = \left[\begin{array}{cc} 1 & 3 \\ 2 & 6 \end{array} \right] \xrightarrow{\text{G.E.}} \left[\begin{array}{cc} 1 & 3 \\ 0 & 0 \end{array} \right] \\
\mathbf{C}(A^T) = c\left[\begin{array}{c} 1 \\ 2 \end{array} \right] \qquad \mathbf{N}(A^T) = y_2\left[\begin{array}{c} -3 \\ 1 \end{array} \right]
$$


아래는 위에서 구한 행렬 $A$의 각 주요 부분 공간을 나타낸 것입니다. 

<p align="center"><img src="https://user-images.githubusercontent.com/45377884/92763326-5e5a2500-f3ce-11ea-8b75-06ee6c407bd7.png" alt="ortho_subspace" style="zoom:40%;" /></p>

위 그래프로부터 Column space 와 Left-Null space가 직교 관계에 있으며 $\mathbf{C}(A) \perp \mathbf{N}(A^T)$, Row space와 Null space가 각각 직교 관계에 있는 것 $\mathbf{C}(A^T) \perp \mathbf{N}(A)$ 을 알 수 있습니다.

Column space 와 Left-Null space, Row space 와 Null space 처럼, 두 부분 공간 $V,W \in \mathbb{R}^n$ 이 있을 때, $V \perp W$ 조건이 성립하며 $\text{Dim}(V) + \text{Dim}(W) = n$ 이면 Orthogonal complement(직교 여공간) 관계에 있다고 합니다.