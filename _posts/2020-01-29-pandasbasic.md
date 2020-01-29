---
layout: post
title: Pandas Basic
category: Python
tag: Python
---

 

아래 내용은 [파이썬 머신러닝 완벽 가이드](http://www.yes24.com/Product/Goods/69752484) (권철민, 위키북스, 2019)를 참조하여 작성하였습니다. 아래에 존재하는 'col1', 'col2' ... 등은 가상의 이름입니다. 본인의 데이터에 맞게 변형하시면 됩니다.



## Pandas Basic

- Pandas 모듈 임포트하기

```python
import pandas as pd
```

- DataFrame 생성하기, 외부파일 읽어오기 : .read_csv()

```python
df = pd.read_csv("file_name.csv")
```

- .head(), .tail(), .shape(), .describe(), .value_counts()

```python
df.head()	# .head(), .tail()은 각각 DataFrame의 맨 윗부분, 맨 아랫부분을 출력한다. default == 5
df.tail(3)
df.shape	# DataFrame의 모양 표현 (row의 수, column의 수)가 출력된다.
df.describe()	# DataFrame에 대한 대략적인 정보를 출력한다. [R의 str(), dplyr::glimpse()와 유사]
df.value_counts()	# column 내부에 어떤 value가 몇 개 있는지를 출력해준다.
```



- Column Data set 생성, 수정, 삭제

  - 생성과 수정

  ```python
  df['new_colname1'] = 0
  df['new_colname2'] = df['col1'] + df['col2'] * 10
  ```

  - 삭제

  ```python
  df.drop('col1', axis=1)
  #axis=0 : 행을 삭제, axis=1 : 열을 삭제
  #inplace=True : 원본 데이터를 변형, inplace=False : 원본 데이터는 변형 없이 column이 drop된 DataFrame으로 새로운 객체 생성
  ```

  - Index 객체

  ```python
  index_arr = df.index	#Index 객체를 추출한다.
  index_arr.values	#ndarray 형태로 반환한다.
  ```

  

- __데이터 셀렉션 및 필터링__

  - [] 연산자

  ```python
  #[] 내부에는 column 이름 혹은 표현식이 들어가야 한다.
  df['col2']
  df[0:3]
  ```

  - __iloc[] 연산자__ : Row와 Column의 index순서(숫자)로 추출하기

  ```python
  df.iloc[0, 1]	#df[row_index, col_index]
  ```

  - __loc[] 연산자__ : Row의 index value와 Column의 이름으로 추출하기

  ```python
  df.loc[1,'col2']	#df['index_value', 'colname']
  df2.loc['a', 'col3']	#df2의 index가 'a','b','c' ... 일 때
  ```

  - 불린 인덱싱

  ```python
  df_boolean = df[df['col1'] > 5]
  df[df['col1'] > 5][['col1','col2']]
  ```



- 정렬, 결손 데이터 처리하기

  - .sort_values()

  ```python
  df.sort_values(by=['col1'])	#디폴트 값은 ascending = True.
  df.sort_values(by=['col1', 'col2'], ascending=False)
  ```

  - .isna()

  ```python
  df.isna()	#NaN 값을 True, 아닌 값을 False로 불린 인덱싱
  df.isna().sum()	#각 column에 있는 True의 개수(NaN값의 개수) 반환
  ```

  - .fillna()

  ```python
  df['col4'] = titanic_df['col4'].fillna('val1')
  #col4에 있는 NaN 값을 val로 대체
  #이후 df['col4'].isna().sum() 실행시 0이 나오게 된다. 
  ```



- apply lambda 식으로 데이터 가공

  - lambda 함수 만들기

  ```python
  def get_sqr(num):
      return num ** 2		#인 함수를 lambda 함수로 바꾸면
  
  lambda_sqr = lambda x:x ** 2	#가 된다.
  ```

  - map() 함수와의 결합

  ```python
  list_a = [1, 2, 3]
  squares = map(lambda x : x ** 2, list_a)
  list(squares)
  > [1, 4, 9]
  ```

  - Pandas 에서 apply()로 lambda 결합하기

  ```python
  df['new_col1'] = df['col1'].apply(lambda x : len(x))
  df[['col1', 'new_col1']]
  
  df['new_col2'] = df['col2'].apply(lambda x : 'val2' if x < 10 else 'val3')
  df[['col2'], ['new_col2']]
  ```

  