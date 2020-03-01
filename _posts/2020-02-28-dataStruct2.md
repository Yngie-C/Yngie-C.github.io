---
layout: post
title: 2. 자료구조 구현을 위한 C 프로그래밍 기법
category: Data Structure with C
tag: Data-structure
---



## 1) 배열

- 배열의 개념 : 다수의 데이터를 저장하고 처리하는 경우, 유용하게 사용할 수 있는 것이 배열

- 1차원 배열

  - 1차원 배열의 선언 : 필요한 것은 배열이름, 자료형, 길이정보 이다.

  ```c
  int oneDimArr[4];
  /// int : 배열을 이루는 요소(변수)의 자료형
  /// oneDimArr : 배열의 이름
  /// [4] : 배열의 길이
  /// 즉, int형 변수 4개로 이루어진 배열을 선언하되, 그 배열의 이름은 oneDimArr.
  ```

  - 선언된 1차원 배열에의 접근

  ```c
  #include <stdio.h>
  
  int main(void)
  {
      int arr[5];
      // 길이가 5이고, int 자료형이 들어가는 배열 arr 생성
      int sum=0, i;
  
      arr[0]=10, arr[1]=20, arr[2]=30, arr[3]=40, arr[4]=50;
      // 배열의 각 자리에 값을 저장
  
      for (i = 0; i < 5; i++)
      {
          sum += arr[i];
      }
      // 배열 요소에 저장된 값을 전부 더함
  
      printf("배열 요소에 저장된 값의 합: %d \n", sum);
      return 0;
      
  }
  ```

  위 코드처럼 `for` 문을 통해서도 배열의 모든 요소에 순차적으로 접근할 수 있다.

  - 1차원 배열의 초기화 : 기본 자료형 변수들은 선언과 동시에 초기화를 할 수 있다. 초기화의 방법은 총 세가지다.
    1. `int arr1[5] = {1, 2, 3, 4, 5};` - 중괄호 내에 초기화할 값을 입력하면, 이 값들이 순서대로 저장된다.
    2. `int arr2[] = {1, 2, 3, 4, 5, 6, 7};` - 두 번째 예에서는 배열의 길이를 나타내는 부분이 비어있다. 하지만 컴파일러가 자동으로 초기화 리스트의 수를 참조하여 길이 정보를 채워주게 된다.
    3. `int arr1[5] = {1, 2};` - 배열의 길이보다 초기화 리스트가 짧을 경우 채울 값이 없는 요소의 자리에는 자동으로 0이 채워지게 된다.

  ```c
  #include <stdio.h>
  
  int main(void)
  {
      int arr1[5] = {1, 2, 3, 4, 5};
      int arr2[ ] = {1, 2, 3, 4, 5, 6, 7};
      int arr3[ ] = {1, 2};
      int ar1Len, ar2Len, ar3Len, i;
  
      printf("arr1의 크기: %d \n", sizeof(arr1));
      printf("arr2의 크기: %d \n", sizeof(arr2));
      printf("arr3의 크기: %d \n", sizeof(arr3));
  	// sizeof 연산에 배열 이름을 넣어주면 '바이트 단위의 배열 크기'를 반환
      
      ar1Len = sizeof(arr1) / sizeof(int); // arr1의 길이 계산
      ar2Len = sizeof(arr2) / sizeof(int); // arr2의 길이 계산
      ar3Len = sizeof(arr3) / sizeof(int); // arr3의 길이 계산
      //각 배열의 크기(sizeof)를 배열을 구성하고 있는 자료형의 크기로 나누어준다.
  
      for (i = 0; i < ar1Len; i++)
      {
          printf("%d ", arr1[i]);
      }
      printf("\n");
      
      for (i = 0; i < ar2Len; i++)
      {
          printf("%d ", arr2[i]);
      }
      printf("\n");
      
      for (i = 0; i < ar3Len; i++)
      {
          printf("%d ", arr3[i]);
      }
      printf("\n");
      return 0;
  }
  ```

  

  - 문자 배열

    - char형 배열의 문자열 저장과 Null 문자

    C언어에서는 큰 따옴표를 이용하여 문자열을 표현한다. 다음과 같이 문자열을 구성하면 배열에 문자열이 저장된다.

    ```c
    char str[14] = "Good Morning!";
    char str[ ] = "Good Morning!";
    ```

    위 코드의 2번째 줄과 같이 배열의 길이를 생략해주어도 자동으로 삽입된다. 하지만 (1, 'G'), (2, 'o'), (3, 'o'), (4, 'd'), (5, ' '), (6, 'M'), (7, 'o'), (8, 'r'), (9, 'n'), (10, 'i'), (11, 'n'), (12, 'g'), (13, '!') 에서 볼 수 있듯 총 13개의 문자가 있는데 배열의 길이는 14이다. 이렇게 문자 배열을 만들 때 배열 길이가 +1이 되는 이유는 마지막에 '\0'이라는 문자가 자동으로 삽입되기 때문이다. 여기서 끝에 자동으로 삽입되는 '\0' 을 가리켜 **Null** 문자라 한다. 

    ```c
    #include <stdio.h>
    
    int main(void)
    {
        char str[] = "Good Morning!";
        //char형 배열을 선언하고 이를 문자열로 초기화. 배열의 길이는 알아서 결정된다.
        
        printf("배열 str의 크기: %d \n", sizeof(str));	//배열의 크기를 출력
        printf("Null 문자 문자형 출력: %c \n", str[13]);	//Null 문자를 문자형으로 출력
        printf("Null 문자 정수형 출력: %d \n", str[13]);
    
        str[12]='?';	// 배열 str에 저장된 문자 중 12번째 문자(!)를 바꾸기 위한 코드
        printf("문자열 출력: %s \n", str);
        return 0;
    }
    
    >>>
    배열 str의 크기: 14
    Null 문자 문자형 출력:  
    Null 문자 정수형 출력: 0
    문자열 출력: Good Morning?
    ```

    이로부터 Null 문자의 아스키 코드 값은 0이며, 이를 문자의 형태로 출력할 경우 아무런 출력이 발생하지 않는다는 것을 알 수 있다.

    

    - Null 문자와 공백 문자 구별하기

    ```c
    #include <stdio.h>
    
    int main(void)
    {
        char nu = '\0';
        char sp = ' ';
        printf("%d %d", nu, sp);
        return 0;
    }
    >>>
    0 32
    ```

    이를 통해 Null 문자의 아스키 코드 값은 0이며 공백 문자의 아스키 코드 값은 32임을 알 수 있다.

    

    - `scanf` 함수를 이용한 문자열의 입력

    `scanf` 함수를 통해서도 배열에 문자열을 입력받을 수 있다. 아래의 코드를 보자.

    ```c
    #include <stdio.h>
    
    int main(void)
    {
        char str[50];
        int idx=0;
    
        printf("문자열 입력: ");
        scanf("%s", str);
        //일반적으로 scanf 함수 호출문 구성시, 데이터를 입력받을 변수의 이름 앞에는 &를 붙이지만 		문자열은 &를 붙이지 않는다.
        printf("입력받은 문자열: %s \n", str);
    
        printf("문자 단위 출력: ");
        while(str[idx] != '\0')
        //scanf로 입력받은 문자열 마지막에도 Null 문자가 삽입되어 있다.
        {
            printf("%c", str[idx]);
            idx++;
        }
        printf("\n");
        return 0;
    }
    ```

    위 코드에서 볼 수 있듯, C언어에서 표현하는 모든 문자열의 끝에는 Null 문자가 자동으로 삽입된다.  즉, Null 문자가 있으면 문자열이고, 없으면 문자열이 아니다. 하지만 위 코드에서 `scanf` 함수에 공백이 있는 문자열(예를 들면, "I like C programming")을 입력할 경우 `scanf` 함수는 공백 문자를 기준으로 "I", "like", "C", "programming" 이라는 4개의 문자열이 입력된 것으로 인식한다. 때문에 `scanf` 는 문자열을 입력받기에는 적절하지 않다. 

    ```c
    char arr1[] = {'H', 'i', '~'}
    char arr2[] = {'H', 'i', '~', '\0'}
    ```

    위의 예에서도 배열 arr1은 마지막에 Null 문자가 없으므로 단지 문자가 저장된 배열일 뿐이다. 그에 비해 배열 arr2는 Null 문자가 마지막에 존재하므로 문자열이 저장된 배열이 된다.

    

    - 문자열의 끝에 Null 문자가 필요한 이유

    문자열의 끝에 Null 문자가 없다면 앞선 예시처럼 `while` 문을 이용해 문자열을 출력하지 못한다. 즉, 문자와 문자열의 경계가 사라진다. 그래서 Null 문자를 이용해서 문자열의 끝을 표시하는 것이다. 아래의 코드를 보자.

    ```c
    #include <stdio.h>
    
    int main(void)
    {
        char str[50] = "I like C programming";
        printf("string: %s \n", str);
    
        str[8] = '\0';  //9번째 요소에 Null 문자 저장
        printf("string: %s \n", str);
    
        str[6] = '\0';  //7번째 요소에 Null 문자 저장
        printf("string: %s \n", str);
    
        str[1] = '\0';  //2번째 요소에 Null 문자 저장
        printf("string: %s \n", str);
    
        return 0;
    }
    >>>
    string: I like C programming
    string: I like C
    string: I like
    string: I
    ```

    위 코드에서 볼 수 있듯 `printf` 함수도 Null 문자를 기준으로 문자열을 구분한다. 

    

- 다차원 배열
  - 다차원 배열의 선언 : 2차원과 3차원 배열의 선언 형태는 다음과 같다
  
  ```c
  int arrTwoDim[5][5];
  int arrThreeDim[3][3][3];
  ```
  
  2차원 배열 선언시 `int arrTwoDim[a][b];` 에서 a는 세로 길이(행의 개수), b는 가로 길이(열의 개수)를 나타낸다. 2차원 배열도 `sizeof` 함수를 통해서 계산해낼 수 있다.
  
  ```c
  #include <stdio.h>
  
  int main(void) {
    int arr1[3][4];
    int arr2[7][9];
  
    printf("세로 3, 가로 4: %d \n", sizeof(arr1));
    printf("세로 7, 가로 9: %d \n", sizeof(arr2));
    return 0;
  }
  >>>
  세로 3, 가로 4 : 48
  세로 7, 가로 9 : 252
  ```
  
  
  
  - 다차원 배열의 초기화
  - 문자 다차원 배열

<br/>

## 2) 포인터

- 포인터 개념
- 포인터 선언
- 포인터 연산
  - 주소 연산자(&)
    
  - 참조 연산자(*)
  
- 포인터 초기화
- 포인터와 문자열
- 포인터 배열
- 포인터의 포인터(이중 포인터)

<br/>

## 3) 구조체

- 구조체 개념
- 구조체 선언
- 구조체 변수의 초기화
- 데이터 항목의 참조



<br/>

## 4) 재귀호출

- 재귀호출의 개념
- 재귀호출의 예
- 재귀호출의 예2

<br/>