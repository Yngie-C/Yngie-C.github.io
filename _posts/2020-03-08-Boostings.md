---
layout: post
title: XGBoost, LightGBM and Catboost
category: Machine Learning
tag: Machine-Learning
---





본 포스트는 참고도서인 핸즈 온 머신러닝에 없는 내용이지만 각종 대회에서 많이 쓰이고 있는 앙상블 기법인 XGBoost와 LightGBM에 대한 내용을 담고 있습니다. 본 포스트의 내용은 주로 [파이썬 머신러닝 완벽 가이드](http://www.yes24.com/Product/Goods/69752484) 를 참고하였습니다.



## 1) XGBoost(XGB)

- **XGBoost(eXtra Gradient Boost)** : 그래디언트 부스팅 기반의 ML패키지 중 하나. GBM에 기반하고 있으며, GBM이 느리고 과적합을 막기위한 규제가 없다는 문제점을 해결.
- XGBoost의 주요 **장점**

| 항목                         | 설명                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| 뛰어난 예측 성능             | CART기반 알고리즘으로 분류와 회귀 영역 모두에서 사용할 수 있으며 뛰어난 예측 성능을 발휘한다. |
| GBM 대비 빠른 수행 시간      | 병렬 수행이 가능하고 다양한 기능을 제공하기 때문에 GBM대비 속도가 빠르다. 다만 GBM대비 상대적인 것이지 다른 머신러닝 알고리즘과 비교하면 느린편. |
| 과적합 규제                  | 자체 과적합 규제 기능이 있기 때문에 과적합에 좀 더 강하다.   |
| Tree pruning (나무 가지치기) | max_depth 파라미터 이외에도, tree pruning을 통해 더 이상 긍정 이득이 없는 분할을 가지치기 하여 분할 수를 더 줄인다. |
| 내장된 교차 검증             | 반복 수행시마다 내부적으로 교차 검증을 시행하며, 이를 바탕으로 조기 중단(Early Stopping)을 할 수 있는 기능도 내장하고 있다. |
| 결손값 자체 처리             | 결손값을 자체 처리할 수 있다.                                |

- XGBoost 하이퍼 파라미터 : 기본적으로는 GBM과 유사한 파라미터를 가지고 있으며 여기에 Early stopping과 규제를 위한 하이퍼 파라미터가 추가되었다.
  - 주요 일반 파라미터
    - `booster` : tree based model인 gbtree 와 linear model인 gblinear 중 선택 (default=gbtree)
    - `silent` : 출력 메시지를 나타낼 것인지(1인 경우 메시지 숨김, default=0)
    - `nthread` : CPU의 실행 스레드 개수를 설정. 일부 스레드만 사용하고 싶을 때 변경 (default=전부 다 사용)
  - 주요 부스터 파라미터
    - `eta [default=0.3, alias: learning rate]` : 학습률(learning rate)과 같은 역할을 하는 파라미터. 0에서 1 사이의 값을 지정한다. 모델의 성능을 좌우하는 주요 파라미터
    - `num_boost_rounds` : GBM의 n_estimator와 같은 파라미터. 몇 개의 트리를 사용할 것인지를 결정
    - `min_child_weight [default=1]` : GBM의 min_child_leaf와 유사하며 과적합을 조정하기 위해서 사용한다.
    - `gamma [default=0, alias: min_split_loss]` : 트리의 리프 노드를 추가적으로 나눌지를 결정할 최소 손실 감소 값. 해당 값보다 큰 손실이 감소된 경우에만 리프 노드를 분리한다. 값이 클수록 과적합 효과가 있다.
    - `max_depth [default=6]` : 트리 기반 알고리즘의 max_depth와 동일하다. 트리의 최대 깊이를 설정한다. 깊이 제한을 해제하려면 0으로 설정한다. 이 또한 모델의 성능을 좌우하는 주요 파라미터이다.
    - `sub_sample [default=1]` : 트리가 샘플링되는 비율을 지정. 0부터 1사이의 값을 사용가능하나 일반적으로는 0.5~1의 값을 사용한다.
    - `colsample_bytree [default=1]` : GBM의 max_feaures와 유사. 트리 생성에 필요한 피처(feature)를 임의로 샘플링하는 데 사용. 피처가 많은 경우 과적합을 줄이기 위해 사용한다.
    - `lambda [default=1, alias: reg_lambda]` : L2 Regulation 적용값.
    - `alpha [default=0, alias: reg_alpha]` : L1 Regulation 적용값.
    - `scale_pos_weight [default=1]` : 특정 값으로 치우친 비대칭 클래스로 구성된 데이터 세트의 균형을 유지하기 위한 파라미터



<br/>

## 2) LightGBM(LGBM)

- **LightGBM(Light Gradient Boosting Machine)** : 여전히 오래 걸리는 XGBoost의 학습 시간을 개선하기 위해 개발된 부스팅 계열의 알고리즘. 학습 시간이 짧고 메모리 사용량이 적어 하이퍼파라미터 최적화를 위한 학습을 많이 시도할 수 있다는 것이 장점이다. 반면에 데이터 세트가 적을 경우에 과적합이 발생할 수 있다는 단점이 있다.(일반적으로 10,000건 이하의 데이터) LGBM에서는 범주형 피처의 자동 변형과 최적 분할을 제공한다.(원-핫 인코딩 등의 과정 없이도 노드 분할 가능)



- 기존 트리 기반 알고리즘과의 차이점 : **리프 중심의 트리 분할** 사용
  - 균형 트리 분할(Level Wise) 방식 : 최대한 균형 잡힌 트리를 생성. 생성되는 트리의 깊이를 맞춘다. 트리의 깊이를 최소화함으로써 오버피팅을 방지할 수 있다는 장점이 있지만 균형을 맞추는 데 많은 시간이 들게되어 학습 시간이 오래 걸린다. GBM과 XGBoost 모두 이 방식의 분할 방식을 사용한다.
  - 리프 중심 트리 분할(Leaf Wise) 방식 : LGBM이 트리를 분할하는 방식이다. 최대 손실(max delta loss)이 발생하는 노드라면 균형이 맞지 않더라도 분할한다. 결과적으로 비대칭적인 트리가 형성되지만 손실을 최소화할 수 있다. 

![Level_vs_Leaf](https://i0.wp.com/mlexplained.com/wp-content/uploads/2018/01/DecisionTrees_3_thumb.png?w=1024&ssl=1)



- LightGBM 하이퍼파라미터
  - 주요 파라미터
    - `num_iterations [default=100]` : 반복 수행하려는 트리의 개수를 지정. GBM의 n_estimator와 같은 역할을 한다.
    - `learning_rate [default=0.1]` : 학습률. 작게 할수록 학습 시간이 길어지며, 너무 크게 하면 최적값을 찾지 못한다.
    - `max_depth [default=-1]` : 트리의 최대 깊이. 과적합을 막기 위해서 숫자를 적절하게 낮추는 것이 좋다. 하지만 LGBM은 리프 중심의 트리 분할 방식을 사용하므로 이전의 방법보다는 좀 더 크게 설정해주는 것이 일반적이다.
    - `min_data_in_leaf [default=20]` : GBM의 min_sample_leaf와 같은 파라미터이다. 최종 결정 클래스인 리프 노드가 되기 위해서 최소한으로 필요한 노드의 수를 가리킨다.
    - `num_leaves [default=31]` : 하나의 트리가 가질 수 있는 최대 리프 개수이다.
    - `boosting [default=gbdt]` : 부스팅의 트리를 생성하는 알고리즘을 기술한다. (gbdt=일반적인 그래디언트 부스팅 트리, rf=랜덤 포레스트)
    - `bagging_fraction [default=1.0]` : 
    - `feature_fraction [default=1.0]` :
    - `lambda_l2 [default=0.0]` : 
    - `lambda_l1 [default=0.0]` : 
  - Learning Task 파라미터
    - objective
  - 하이퍼 파라미터 튜닝 방안 : 



</br>

## 3) Catboost

- Catboost : Catboost 또한 XGBoost의 학습 시간을 줄이기 위한 알고리즘이다. 