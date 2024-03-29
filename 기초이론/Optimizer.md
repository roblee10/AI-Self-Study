# Optimizer

# Gradient Descent

- θ를 미지수로 갖는 목적함수 $J
(
θ
)$를 최소화 하는 방법
- 모델의 예측값과 실제 결과값의 차이를 줄이기 위한 방법 중 하나가 Gradient Descent이다. 이 방법은 목적함수 $J(θ)$를 최소화하는 방법으로, 다음과 같은 식을 사용한다.

$$

θ_{t+1}=θ_{t}−η∇_{θ}J(θ)

$$

여기서 $θ$ 는 미지수이고, $η$ 는 학습률(learning rate)을 의미한다. Gradient Descent는 머신러닝에서 가장 기본적인 최적화 알고리즘 중 하나이며, 다양한 변형 기법이 존재한다.

### **Stochastic Gradient Descent**

- 문제점: Gradient Descent 시간 오래걸림
- 방법: 한번의 파라미터 업데이트를 위해 하나의 훈련 데이터를 사용

$$
θt+1=θt−η∇θJ(θ;x(i),y(i))
$$

- 효과: 빠른 업데이트
- 한계: 큰 분산으로 수렴 방해될 수 있음
- 사용법
    - **`optimizer** **=** **torch.optim.SGD(model.parameters(),** **lr=learning_rate)**`

# ADAM(Adaptive Moment Estimation Optimizer)

- 모멘텀과 RMSProp를 합친 개념

### 모멘텀(관성) optimizer

- 문제점: local minimum을 탈출해야 함
- 방법: 현재 Parameter를 업데이트 할 때 이전 gradient들도 계산에 포함해주면서 업데이트.
    - 현재 gradient가 0이더라도 이전 gradient(관성) 으로 인해 나아가게됨
- 효과 : local minimum 탈출
- 한계점: 수렴지점에서 요동치게 됨

### Adagrad

- 문제점: SGD, Momentum 은 모든 Parameter에 대해 같은 learning rate 적용
- 방법: Adagrad 는 각 Parameter의 업데이트 빈도 수에 따라 업데이트 크기를 다르게 해줌
- Ada = Adaptive 를 뜻함
- 효과 : 상대적으로 조금 업데이트 된 Parameter를 수렴점에 빨리 도달하도록 크게 업데이트 해줄 수 있다.
- 한계점: 파라미터의 t(업데이트 횟수) 가 증가할 수록 learning rate 가 소실되는 문제 발생

### Adadelta

- 문제점: Adagrad의 learning rate 소실 문제 해결
- 방법:
    - $w$ 인 window를 사용하여 지난 w 개의 gradient만 저장
    - Adagrad와 다르게 gradient 제곱의 합을 저장하지 않고 gradient의 제곱에 대한 기댓값 $E[g^2]t$ 저장
    - 과거 gradient 정보의 영향력 감소를 위한 식 사용

### RMSProp

- Adadelta와 유사

### Adam(Adaptive Moment Estimation)

- 방법: 각 파라미터마다 다른 크기의 업데이트 적용
    - Adadelta와 유사하게 gradient 기댓값 $E[g]t$ 활용
    - 이 부분이 momentum을 의미