9/22

# Machine Learning

### 머신러닝

- 우리들이 가지고 있는 데이터로부터 명시적인 프로그래밍 없이 머신(Machine)이 배운다.(Learning)
- 연구 분야이며 기술이 아니라 학문이다.

#### 머신러닝 왜 쓸까?

- 예측 또는 추론
- 데이터 분석, 즉 데이터의 패턴을 찾는 것.
  - 학습(트레이닝)

#### 머신러닝 딥러닝?

- 추론을 트레이닝한 결과
- 성능 측정 방식이 머신러닝은 잘 맞췄다 라는 것을 지표로 삼고, 딥러닝은 잘 못맞춘 모델을 지표로 삼는다.

#### Google Colab

- 구글의 tpu, gpu사용가능
- 웹으로 연습하기에는 최적
- 긴 시간의 작업이 필요한 경우에는 좋지않다.
- 대용량 데이터로 연습할 때 gpu를 쓰면 된다. colab의 장점.

#### 프로그래밍으로 스팸필터 만들기

- 타이틀에서 발견되는 단어들을 조건문으로 걸러낼 것인데 매번 그렇게 걸러낼 것인가?
- 또는 예외가 생긴다면 어떻게 대처 할 것인가.

#### 머신러닝으로 스팸 필터 만들기

- 1. **모델은 데이터에 맞게 준비해야한다.**

- 2. 알고리즘 훈련

- 하이퍼 파라미터

  - 사람이 직접 알고리즘에 설정해 주는 값

- 3. 검증 

  - 만족하지 않은 결과라면 계속해서 데이터를 추가하고 훈련을 한다.

- 모델 평가는 오차 기준이 0인것이 결코 좋은 것이 아니다.

- 그렇게 데이터의 패턴을 찾아낸다 -> 데이터 마이닝 -> 머신러닝

  - 대용량 데이터로부터 데이터의 패턴을 찾아내는 과정을 데이터 마이닝이라고 합니다.

#### 대표적인 머신러닝 시스템

1. 지도학습

   1. 훈련데이터와 레이블을 모두 사용한다.
   2. 분류와 회귀에 사용됨

2. 비지도 학습

   1. 훈련 데이터만 사용함
   2. 군집, 시각화, 이상치 탐지, 차원 축소등에 사용됩니다.
   3. 시각화가 들어간 이유....
   4. 차원 축소라는 건 중요한 데이터만 볼 수 있도록 만들어 내는 과정.
      1. 결과에 영향을 미치는 데이터만 선별하는 것.

3. 준지도 학습

   1. 지도 학습과 비지도 학습을 모두 사용한다.
   2. 보통 첫번째는 비지도 학습을 사용하고 두번째로 지도 학습을 사용한다.
   3. 페북이 얼굴을 스캔해 태그하지 않은 친구가 태그되어있다.

4. 강화 학습

   1. 패널티와 보상.
   2. 컴퓨터가 스스로 보상과 벌점의 개념을 활용해 학습한다.
   3. 대표적인 예시로 게임 A.I.가 있다.

   

### 1. 지도학습

- 문제와 정답을 주고 컴퓨터가 이를 학습함.

- 보통 분류(Classification)와 회귀(Regression)문제를 다룬다.

- One-Hot 인코딩

  - 하나에만 집중하겠다. 내가 원하는 타입으로 인코딩해서.

  - 청바지 -> 0

  - 반바지 -> 1

  - 치마 -> 2

    |  청  |  반  |  치  |
    | :--: | :--: | :--: |
    |  1   |  0   |  0   |
    |  0   |  1   |  0   |
    |  0   |  0   |  1   |

- 분류는 고양이와 강아지, 스팸인지 아닌지 등을 구분해 내는 문제이다.

  - 0, 1, 2 같은 모습 -> 분류

- 회귀란 올해 옥수수 수확량, 인구 예측 등 연속적인 값을 예측하는 문제입니다.

  - 실수형태 -> 연속적인 값의 나열

### 2. 비지도 학습

- 컴퓨터에게 문제만 주는 학습 방법

- 연관성 있는 데이터를 묶어내는 군집(Clustering)이 대표적이다.

- 데이터에 따라 다르게 표현하는 시각화(Visualization)

  

### 일반화

- 일반화란 새로운 데이터에도 머신이 잘 적응해야 한다는 개념
- 데이터의 패턴을 과도하게 분석하는 것을 '과대적합'이라한다.
- 데이터의 면면성을 놓쳐 면밀하게 분석하지 못하는 것을 '과소적합'이라한다.
- 누구나 공감할 수 있는 결과를 낼 수 있도록 데이터를 끄집어 내는것.
- '특성공학'이 되고 안되고의 차이가 크다.
- 하이퍼 파라미터에 값을 준다는 것이 일반화의 '과정'이라고 볼 수 있다.

====================================================================================

이후 부터  Colab에 필기되어있음.



====================================================================================

9/23

## 사이킷 런의 알고리즘 모델

- 추정기(BaseEstimator)
- 변환기(Transformer)
- fit : 적용(훈련) -> 알고리즘에 데이터를 적용시킨다고 보면됨.
- predict : 예측
- fit-transform
  - 적용할 기준과 적용될 데이터.



### 변환기를 만들기

- 개인이 사용하는 개인적인 변환기를 만들어 놓는 것이 권장사항이다.

#### PipeLine

[작업, 작업, 작업....]

1. housingDataSets의 특성 조합
2. 결측치 수정 및 삭제
3. 변환.



## 수학

#### 함수

##### 측정함수

- 스코어링
  - 높으면 좋다
  - 보통 머신러닝
- 손실
  - 낮으면 좋다
  - 딥러닝

##### 회귀분석에서 사용되는 성능 지표 함수

- 둘다 성능 측정기
- 낮을 수록 좋다.

- MSE (Mean squre error)
  - 주로 쓰인다.

<img src="/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-23 11.10.59.png" alt="스크린샷 2020-09-23 11.10.59" style="zoom:50%;" /> -> MSE <img src="/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-23 11.14.12.png" alt="스크린샷 2020-09-23 11.14.12" style="zoom:50%;" /> -> RMSE

<img src="/Users/kimhyuk/Downloads/IMG_4087.jpg" alt="IMG_4087" style="zoom: 25%;" />

- m은 전체 개수
- 전체 개수 분에 오차합의 제곱
- y는 target
- Y^는 모델이 예측한 값
- L2 노름
- 정답에서 예측을 빼고 제곱함.
- 유클라디안 거리라고도 부름

- MAE (이상치 많을 때)
  - 잘 안쓰임.
  - 절대값이 들어감.
  - 오차 거리가 수직또는 수평으로만 된다.
  - L1 노름 맨하튼 거리

<img src="/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-23 11.30.56.png" alt="스크린샷 2020-09-23 11.30.56" style="zoom:50%;" />

###### 패널티

- 오차를 최대한 줄여서 패널티를 줄이는게 목적으로 MSE를 주로 사용한다.
- 미분이 가능하기 때문

###### 성능 지표

- 1 - 오차율



## 하이퍼 파라미터

- 사람이 개입한 상태에서 모델이 스스로 알아내는 것.
- 사람이 하는것이 마음에 안들어 만든 것이 그리드 탐색이다.

### 그리드 탐색

- 그리드 서치와 교차검증을 같이함.

========================================================================================

이후 부터  Colab에 필기되어있음.



========================================================================================

9/24

## 총정리

1. 수집 : 크롤링 설문조사 기존DB
   1. 적재
   2. 저장
2. 전처리 : Dtype 지정, Token, 이상치 파악, 결측치 파악
   1. 데이터 마트 구축
3. 데이터 분석 : info(), summary, 비즈니스 파악(특성 조합), 산포도, 시퀀스(시간에 따른 데이터의 변화 파악), 상관관계 파악
4. 데이터 편집 : 변환, 파이프 라인 구축
   1. 트레인 세트 기준 : 이상치 처리, 결측치 처리, 스케일링, 인코딩, 특성 조합
      1. 테스트 셋에도 동일한 세팅으로 적용
      2. 파이프 라인으로 관리하면 편하다.
   2. 트레인, 테스트 나누기 (보통의 비율)
      1. 8 : 2
      2. 7.5 : 2.5
      3. 7 : 3
5. 모델 선정 및 훈련 : GridSearch CV
   1. 베스트 모델 찾기
6. 테스트 데이터를 가지고 훈련.
7. 배포
8. 유지보수
   1. 4 ~ 6단계를 반복함.

### 수업 

- 0, 1, 2는 카테고리이다. 그런데 머신러닝에서는 보통 클래스라고 부른다.
- 분석 문제를 잘 고려해야함 분류? 회귀?

#### 과대 적합 과소적합

과대적합

- 훈련데이터에는 잘 나오지만 검증에서는 성능이 크게 떨어짐
- 데이터를 복잡하게 분석하면 과대적합이 나온다.

과소적합

- 모델의 신뢰도가 훈련이든 검증이든 떨어짐.
- 데이터의 다양성을 잡아내지 못하는 규칙을 적용하면 과소적합이 나온다.



## k-NN

- 데이터 포인트의 가장 가까운 '최근접 이웃'을 찾습니다.

- 성능은 이웃의 갯수가 없을 수록 정확도가 떨어진다.

- 점의 개수와 그 거리를 계산해 관계를 나누는 것.

- 이웃의 개수가 매우 중요, 복잡도에 영향을 미침.

- 그런데 이웃의 개수가 4개이고 그 가운데 테스트 데이터가 찍힌다면, 그 때부터 유클라디안 거리로 계산한다.

  ![스크린샷 2020-09-24 10.07.43](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-24 10.07.43.png)

- 이웃이 적을 때 복잡도가 제일 높다 중앙의 세모와 왼쪽 상단의 파란점은 버려야할 데이터라고 생각하는게 좋다.

![스크린샷 2020-09-24 10.08.55](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-24 10.08.55.png)

- 9 이웃으로 가면 갈 수록 경계면이 완만해진다. 복잡도가 내려갔다는 소리.



#### k-NN 회귀

![스크린샷 2020-09-24 10.40.36](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-24 10.40.36.png)



## 선형 회귀

- k-NN은 이웃의 개수에 따라 예측을 했었는데 선형모델은 '선형대수학'을 사용한다.
- y = 2x + 1
- 2는 기울기 1은 절편
- 기울기와 절편을 구하는게 선형회귀의 '목적'이다.
- "기울기와 절편을 어떻게 조절해야 오차가 제일 적을까."를 머신이 해준다.

![스크린샷 2020-09-24 11.15.08](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-24 11.15.08.png)

- w는 weight -> 가중치
- b는 bias -> 기울기
- w와 b에 의해서 성능이 좌우된다.
- 그럼 그 성능에 대한 지표가 필요하다.
  - 그 지표를 MSE로 삼는다. -> 평균 제곱 오차

![스크린샷 2020-09-24 11.25.59](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-24 11.25.59.png)

- MSE는 오차를 보기 위한.
- U모양으로 그려짐.
- y축은 오차 x축은 w가 된다.
- w에 대한 오차 그래프가 됨.
- 제곱 시키는이유는 음수가 아니기 위해서(오차율이니) 그리고 미분하기 위해서.

- 미분
  - 변화량
  - 예시는 1km를 1시간 동안 걸었을 때 
  - 미래의 값에서 현재의 값을 뺀.
  - 시간을 줄이다 보면 시점이 등장한다.
  - 순간적인 변화량?
  - 기울기 : x가 변했을 때 y가 얼마나 변했냐.
  - 미분 :  x가 순간적으로 얼마나 변했냐.
  - Dw와 w의 오차를 줄이다보면 w의 변화량을 만날 수 있음.
  - 내가 얼마나 움직여야할지 방향까지 알 수 있다.
  - w의 값을 줄이다보면(경사하강법)
  - 어디로 내려가야할지를 알 수 있다.
  - 내려가는 그 차이를 학습률이라고 한다.
  - 학습률을 조정해서 미분을 함
- 그 오차를 최소한 시키는 것을 위해야한다.
- 기울기에 따라 오차가 바뀌기 때문.
- **점들을 가장 잘 표현할 수 있는 직선을 그린다.**



##  릿지(RIdge)와 라쏘(Lasso) 선형회귀 기법

릿지도 회귀를 위한 선형 모델이므로 최소 제곱법 같은 예측 함수를 사용한다.  MSE

k-NN은 이웃의 갯수에 선형회귀는 가중치(w)에 따라 예측(복잡도)에 영향을 준다.

그 가중치를 바꿀 방법이 없지만 릿지(MSE)와 라쏘(MAE)에는 그것을 보완할 패널티라는 것이 존재함.

### 패널티

- 복잡도를 제어할 수 있다.
- 알파값
  - 10의 지수형태로 사용.





========================================================================================

이후 부터  Colab에 필기되어있음. 

2.ipynb

3.ipynb

==============================

1. 일반화 과대소 적합을 어떻게 피해야할까.
2. k-NN은 잘 안쓰인다. 선형회귀의 공식 y = W(T)X + b 
   1. MSE
   2. 미분
3. 릿지와 라쏘를 통해서 가중치를 조절한다. 



========================================================================================

9/25

## 선형 이진 분류



## 이진 선형 분류

y^ = WX +...... + b > 0 -> 1

y^ = WX +...... + b > 0 -> -1 ,  0



### 결정 경계

- 이진 선형 분류기는 선, 평면, 초평면을 사용하여 두개의 클래스를 구분하는 분류기
- 이 부분은 O야 저 부분은 X야 라고 구분짓는 경계
- 만약에 데이터가 경계에 가까웠을 때 이 데이터는 각 경계에 확실한 데이터라고 말할 수 있을까?
- 그것을 결정하는 것이 '결정 확률'이다. w(가중치)의 조절을 통해.
- b값을 기준으로 데이터 특성에 민감한지 둔감한지 알 수 있다.
- 그래프는 계단식의 형태를 가짐.

### 시그모이드 sigmoid

<img src="/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-25 09.19.29.png" alt="스크린샷 2020-09-25 09.19.29" style="zoom:50%;" />

- 확률을 알 수 있다 0과 1 사이에 0.5의 기준이 생기고 0.5 위아래로 1과 0을 측정한다.
- 그래서 몇퍼센트 확률로 1 또는 0이라는 결과를 도출함.
- 규제를 조절할 수 있는 c값이 있다. 복잡도를 조절할 수 있다는 뜻.
- penalty 인자로 l1, l2를 줄 수도 있다. l1은 라쏘, l2는 릿지.



## 선형회귀 vs 선형 이진 분류

|                           선형회귀                           |                        선형 이진 분류                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| 데이터들을 대표해서 선을 어떻게 잘 그을까 하는 알고리즘. y = WX + b. w를 초기화 해야한다. MSE 평균 제곱 오차를 사용오차 제곱의 합을 구해서 전체 개수로 남은 값오차에 영향을 주는 값은 W, b 보통 W로만 구한다 특성이 여러개면 그만큼 다시 구해야하니까.그럼 W를 어떻게 조절해야할까 그때 나오는 것이 미분이다. W를 어떻게 해야 오차를 줄일지가 나온다. (각 계산하는 포인트들을 LR이라고 하고 오차를 줄이기 위한 값을 학습하기 때문에 LR(Learning Rate)라고 부른다. 그리고 그 방법을 GE(경사 하강법이라 부름)) 그 계산으로 인해서 y = WX + b에서 W가 나온다, 이 과정을 계속 반복해 적절치를 찾아냄. 그 가중치를 직접 줄수 있는게 릿지와 라쏘이다. 가중치에 따라 b값이 바뀐다(동시에 돌아간다고 보면됨), 그래서 선형회귀란 오차율이 가장 적고 데이터들을 대표할 수 있는 선을 찾아내는 것이다. 점을 잘 지나기 위해 선을 긋는다. | y = WX + b > 0 -> 1, y = WX + b < 0 -> -1, 선형회귀에서 비롯된 방법, '점들을 잘 분류하기 위한'. 똑같이 가중치와 편향을 조절한다, 점들을 잘 분류하기 위해 선을 긋는다. 경사 하강법을 사용. 1 또는 0을 구분하기 때문에 계단함수를 사용함, 근데 계산함수를 사용하면 기울기를 계산 할 수 없다. 이 때 확률이 들어가는데 이 때 사용하는 것이 sigmoid함수이다. 이때부터 확률을 알아낼 수 있다. 1, 0.5, 0으로 y값이 구분되게 된다. 계속해서 w와 x가 추가하고 선을 긋고 시그모이드로 와서 계산하게 된다. 이 확률에 의해서 오차 역전파라는 것을 할 수 있다. |



## 다중 클래스 분류

- 이진 분류 알고리즘을 소프트맥스를 사용해 one vs one의 계산 방식을 one vs all(rest)로 확장 시킬 수 있다.

  ![스크린샷 2020-09-25 13.07.51](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-25 13.07.51.png)

- 마찬가지로 소프트 맥스도 1과 0을 도출함.
- X1 -> [0.7, 0.1, 0.2] 
- (전체 합) 안의 값을 더하면 1이 나옴.
- 각각의 클래스가 될 확률을 구함.
- 시그모이드는 이중 분류일 때 씀.
- 많이 사용하진 않음.



## 의사 결정 나무

### 결정 트리(Decision Tree)

지니계수(Gini coefficient)

질문의 끝은 ''결정구간'

#### 과대적합 막기

- 트리 생성을 일찍 중단( **사전 가지치기 ( pre pruning )** )
- 트리를 다 만들고 데이터 포인트가 적은 노드를 삭제하거나 병합 ( **사후 가지치기( post pruning )** )
  - 사이킷런에서 채택한 기능.
- 깊이를 정해주지 않으면 거의 무조건 과대적합된다.
- max_depth = 4

#### 트리의 특성 중요도

- 전체 트리를 살펴 보는 것은 힘든 일입니다. 대신 트리가 어떻게 작동하는지 요약하는 속성들을 사용 할 수 있습니다. 가장 널리 사용되는 속성은 트리를 만드는 결정에 각 특성이 얼마나 중요한지를 평가하는 **특성 중요도** 이 값은 0과 1사이의 숫자로, 각 특성에 대해 0은 전혀 사용되지 않았다는 것을 의미하고, 1은 완벽하게 타깃 클래스를 예측 했다는 뜻이 됩니다.
- 특성의 중요도를 보는데에 트리를 사용해서 '볼 수도'있다.



### 결정 회귀 트리

- 분류만 논했지만 회귀에서도 비슷하게 적용된다.
- 심각한 단점이라면 훈련 데이터의 범위 밖의 포인트에 대해 예측 할 수 없습니다.
- 아직 입력되지 않은 데이터에 대한 정보는 예측이 당연히 불가능하다.
- 그러므로 시계열 데이터에는 사용할 수 없고 하면 안된다. 
  - 시계열데이터(시퀀셜 데이터) -> 순서가 중요한 데이터, 구간별 데이터, 자연어처리 NLP (문자열)

<img src="/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-25 14.05.48.png" alt="스크린샷 2020-09-25 14.05.48" style="zoom:50%;" />



### 장단점과 매개변수

- max_depth
- max_leaf_nodes
- min_sample_leaf
  - 샘플 질문을 강제로 늘리는 것.

보통 질문을 제한하는게 제일 정확하고 신뢰성이 있다. 



### 장단점

#### 장점

모델을 쉽게 시각화 할 수 있으며 비전문가도 이해하기가 쉽다.

데이터의 크기에 구애 받지 않는다.

각 특성이 개별적으로 처리가 되어 일반화나 전처리가 필요하지 않다.

#### 단점

과대 적합이 쉽게 된다.

랜덤포레스트-앙상블 방법을 단일 결정 트리의 대안으로 흔히 사용함.



## 앙상블(Ensemble)

앙상블은 여러 머신러닝 모델을 연결하여 더 강력한 모델은 만드는 데에 그 의의가 있다.

### 랜덤 포레스트

- 의사결정 나무가 여러개 모여있음.

- 왜 랜덤이냐 

  - 각 트리는 모델별로 예측을 잘 할테지만, 과대적합 경향을 가진다는데 기초하며, **서로 다른 방향으로 과대적합된 트리**를 많이 만들면 그 결과를 평균냄으로써 과대적합 양을 줄이자 라는 겁니다.

  ##### 부트스트랩 샘플

  - 매개변수로 n_samples 개의 데이터 포인트 중 무작위로 n_samples 횟수만큼 반복 추출 하게 되는데, 이 때 한 샘플이 여러번 추출 될 수 있다. 결론적으로 트리를 만들어야 할 샘플 자체는 원본 데이터셋 크기와 같지만, 어떤 데이터 포인트는 누락 될 수 있고 어떤 데이터 포인트는 중복되어 들어 갈 수 있습니다.

  

  - 예를들어 ['a','b','c','d']에서 부트스트랩 샘플을 만든다면

    ​	['b','d','d','c']

    ​	['a','d','d','b']

    같은 샘플이 만들어 질 수 있다는 이야기 입니다.

    

  - 숲을 이루는 나무들이 각각 달라지도록 무작위성을 주입함. 그 방식으로는

  - 트리를 생성 할 때 사여ㅛㅇ하는 데이터 포인트(특성)을 무작위로 선택

  - 분할 테스트에서 특성을 무작위로 선택하는 방법

  

  - n_samples 
    - 개수
    - 기본은 원본 데이터셋의 크기(x의 개수)
    - 트리에 사용할 특성 개수
  - max_features
    - 최대 샘플의 개수와 종류
    - 특성을 고를 때 그 종류의 개수이다.
    -  매개변수로 고를 특성 수를 조절할 수 있다.
    - 값이 높으면 트리마다 다양성이 떨어진다.
  - n_estimators는 나무의 개수



### 장단점

#### 단점

- 나무를 만들고 과적합 시키고 훈련을 하는데 시간이 너무 많이 걸린다. 





### 그래디언트 부스팅 회귀 트리

- 이름은 회귀 트리지만 분류와 회귀 모두 사용 가능하다.
- 트리를 여러개 만들어 놓는데 처음 나무가 맞추고 그다음 나무가 맞추고 하는 방식으로 
- 순차적으로 학습한다. 
- 과소적합이 된 트리를 넘겨준다.
- 마지막의 트리값이 결과가 된다.

#### LR

- 학습률이 커지면 보정을 강하게 하기 때문에 복잡한 모델을 만들어낸다. 
- 보통 0.01, 0.1정도로 사용함.



## SVM 서포트 벡터 머신

- 서포트 벡터를 찾는게 가장 중요
- RBF커널을 사용함.(방사선 방정식)
- 서포트 벡터란
  - 서포트 벡터란 결정 경계선에 가장 가까이 있는 각 클래스의 데이터 점

### 선형 모델 (Linear SVC)과 비선형 특성

- 차원수를 들린다. 
- 제곱항을 추가해서 그래프를 줄이고 다시 제곱해서 차원을 내린다.
- 특성이 많아지게 되면 힘든 방법이다.



### 커널 기법 활용하기

- 다항식 커널
- RBF커널



### SVM 매개변수 튜닝

1) 𝛾 매개변수의 역할

- 각 샘플들의 영향력
- 커널 폭의 변수 1~0 사이.
- 가우시안 커널 폭의 역수에 해당.

2) C 매개변수의 역할

- C값이 높아지면 잘못 분류된 샘플을 분류하려고 함.
- 서포트 벡터 선정에 영향을 미침.
- 규제값

### 장단점

#### 단점

- 데이터가 너무나 많으면 정확도가 떨어진다.
- 수학적인 관점에서도 모델을 설명하는것 자체가 어려움.
- 전처리와 매개변수 설정에 신경을 많이 써야 한다.

#### 장점

- 다양한 데이터 셋에 잘 작동한다.

========================================================================================

이후 부터  Colab에 필기되어있음. 

3_Logistic Regression.ipynb



========================================================================================

9/28

## 특성공학

데이터 분석과 특성 공학은 다른 부분이다.

데이터가 학습하기에 좋은 모양으로 만들어 주는 과정?

#### StandardScaler

- 각 특성의 평균을 0으로 분산을 1로 변경함.
- 데이터의 크기를 변경하지는 않고(최솟값과 최대값 크기를 제한하지 않음.)

#### MinMaxScaler

- 모든 특성이 정확하게 0과 1사이에 위치하도록 데이터를 변경함.
- 최솟값과 최대값에 관여함. 
- 크기를 조정한다. 

#### RobustScaler

- 이상치를 분류해 낼 때 사용함
- 평균과 분산 대신 중간값과 사분위 값을 사용함.

#### Normalizer(정규화)

- 반지름이 1인 원을 그릴 수 있다.
- 삼각함수 -> cos유사도
- 데이터가 얼마나 비슷한 가를 알 수 있다.
- 특성의 통계치를 사용하기보다 행마다 각기 정규화가 됨.

========================================================================================

### 데이터 변환

- shape를 무조건 확인해야한다. 

- 오류를 확인하기 위해서

- ```
  (426, 30)
  (143, 30)
  ```

- feature의 개수는 다르면 안되기 때문에

- lable은 스케일링 하면 안된다.

========================================================================================

## One-Hot Encoding

- 문자형 카테고리로 되어있는 형태(데이터)를 머신러닝이 가능한 데이터로 바꾸기 위해서.

#### 연속형 특성

- 실수나 숫자로 되어있다.

#### 범주형 특성(**categorical feature**), 이산형 특성

- 보통 숫자 값이 아니다.
- 이 데이터를 사용하기 위해 원핫인코딩이 필요하다.

### 원핫 인코더 종류

- 사이킷런(one-hot encoder)
  - 숫자형 데이터에만 작동
- 판다스(pd.get_dummies)
  - 단점이라면 아직 숫자형 카테고리에는 작동하지 않는다.
- 비즈니스에 맞게 직접 만든다.

========================================================================================

## 구간분할과 이산화

### 구간분할

- 데이터에 대해서 구간을 나눌때.
- 구간별로 예측을 함.
- 선형회귀때문에 함.
- 특성이 많이 없는데 선형회귀로 예측해야할때 사용함.
- 구간에 따른 값도 준비를 해야한다.

#### 단점

- 0이 들어있는 데이터가 많을 수가 있어 메모리 효율이 조금 떨어 질 수 있다.

========================================================================================

## 다차항 추가

- 특성간의 관계를 더 추가해줄 때

========================================================================================

## 일변량 통계

- 특성과 타깃 사이에 중요한 통계적 관계가 있는지를 계산해준다.
- 깊이 연관되어있는 특성을 자동으로 선택해준다.
- 그래서 각 특성이 독립적으로 평가됨.
- 사이킷런에 두가지의 방식이 있다.
  - SelectKBest - 고정된 K개의 특성을 선택
  - SelectPercentile - 지정된 비율만큼 특성을 선택

========================================================================================

## 모델 기반 선택

- 데이터 스케일링이 잘 되어있어야함.

========================================================================================

## RFE(반복적 특성 선택)

- 시간이 오래 걸림
- 효과가 좋음.

========================================================================================

## 수치변경

숫자가 적은게 양이 많고 많은게 양이 적다.

구간 분할, 다항식, 상호작용은 데이터가 주어진 상황에서 모델의 성능에 큰 영향을 줄 수 있습니다. 특별히 선형 모델이나 나이브 베이즈 모델 같은 덜 복잡한 모델일 경우입니다.

반면에 트리 기반 모델은 스스로 중요한 상호작용을 찾아낼 수 있고 대부분의 경우 데이터를 명시적으로 변환하지 않아도 됩니다. SVM, k-NN, 신경망 같은 모델은 가끔 구간분할, 상호작용, 다항식으로 이득을 볼 수 있지만 선형모델보다는 영향이 그렇게 뚜렷하지는 않습니다.

<img src="/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-28 11.35.36.png" alt="스크린샷 2020-09-28 11.35.36" style="zoom:50%;" /><img src="/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-28 11.35.51.png" alt="스크린샷 2020-09-28 11.35.51" style="zoom:50%;" />

- log를 씌워서 정규 분포형태로 만들 수 있다. 
- log + 1
  - log = 0이 되면 안되기 때문

========================================================================================

## 모델 평가

## 1. 교차 검증(cross validation)

- train 셋트를 검증한다.
- 여러번 나눠서 측정하기 때문에 훨씬 더 안정적인 통계적 평가 방법.

![스크린샷 2020-09-28 13.08.48](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-28 13.08.48.png)

- 이중 가장 많이 사용되는 방법은 k겹 교차 검증 방식이다. 
- fold란 k겹 교차 검증 방식을 통해 나눠진 비슷한 크기의 데이터 세트들을 의미 한다.
- 폴드의 개수는 cv값으로 설정이 되는데 관례적으로 홀수로 설정해준다.

### 장단점

#### 장점

- 모든 폴드가 훈련, 테스트 대상이 되기때문에 훨씬 신뢰있고 안정성있게 모델을 평가 할 수 있다.

#### 단점

- 연산 비용이 늘어난다. 
- cross_val_score는 정말 단순히 모델에 대한 점수만 만들어 지는 것이지 실제 적용할 모델을 만들어내는건 아니다.

​	

### 1. 계층별 k-겹교차 검증

- 데이터가 정렬되어있을 때 보통 사용

![스크린샷 2020-09-28 13.14.56](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-28 13.14.56.png)



### 2. LOOCV ( Leave One Out Cross Validation )

- LOOCV 교차 방식은 폴드 하나에 샘플이 단 한개 뿐인 k-겹 교차 검증이라고 생각 할 수 있습니다. 
- 테스트를 할 때마다 하나의 데이터 포인트를 선택하여 테스트 세트로 사용하게 됩니다. 
- 작은 데이터 셋에는 좋은 결과를 만들어냄.



### 3. 임의 분할 교차 검증 ( Shuffle split cross validation )

- 임의 분할 교차 검증은 훈련 크기(train_size) 만큼의 포인트로 훈련 세트를 만들고, 
- 훈련 세트와 중첩되지 않은 포인트들을 훈련 세트로 만들어 사용합니다. 
- 데이터 크기가 매우 클 때 사용.

![스크린샷 2020-09-28 13.17.44](/Users/kimhyuk/Library/Application Support/typora-user-images/스크린샷 2020-09-28 13.17.44.png)



## 2. 그리드 서치

- 교차 검증과는 다르게 그리드 서치는 모델의 검증이다.
- 하이퍼 파라미터를 찾기 위한 방식이 그리드 서치이다.
- 하이퍼 파라미터 리스트를 만들어 놓고 가장 좋은 결과의 파라미터를 찾아낸다.
- for문으로 하는 약간 무식?한 기법







========================================================================================

이후 부터  Colab에 필기되어있음. 