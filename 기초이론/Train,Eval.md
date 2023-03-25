# Train, Eval

Pytorch는 train mode, eval mode 로 설정 가능

## model.train()

- model 의 layer 들을 train mode 로 바꿔준다.
- Batch Normalization layer 는 Batch statistics 이용
- Dropout Layer 가 확률값에 따라 activate

```python
model.train()
```

### Autograd

- PyTorch 의 automatic differentiation package의 기능
- 뉴럴넷의 backpropagation 중 필요한 gradient 계산을 대신 해준다.

### loss.backward()

- `Require_grad=True` 로 설정된 모든 텐서들의 gradient 계산
- Backpropagation에서 gradient 계산의 시작점이 되는 loss 값부터 .backward()로 미분 연산
- loss 부터 시작 parameter 까지 chain rule 에 따라 미분 연산

```python
criterion = CrossEntropyLoss
loss = criterion(output, labels)
loss.backward()
```

### 변수.grad

- 실제 graidient 값 확인 용도

```python
b.grad # tensor([-3.3881],tensor([-1.9439])
w.grad # tensor([-6.7762],tensor([-3.8878])
```

### .zero_()

- autograd 는 효율성을 위해 gradient 를 자동으로 축적하여 연산
- 따라서 해당 batch 의 gradient update 후 다음으로 넘어가기 전 0 으로 초기화 필요

```python
b.grad.zero_()
w.grad.zero_()
```

### with torch.no_grad()

- Pytorch의 gradient를 계산하기 위한 DCG(Dynamic Computation Graph) 에 영향을 안 주고 일반 tensor 연산을 하도록 한다
- 즉 해당 범위 안에선 gradient를 계산하지 않도록 한다.
- validation에서 자주 사용

```python
with torch.no_grad():
  b -= lr * b.grad
  w -= lr * w.grad
```

## model.eval()

- model 의 layer 들을 evaluation mode 로 바꿔준다
- Batch Normalization layer의 Batch statistics ****대신**** Running Statistics 이용
    - Dropout Layer deactivate, 이런 부분이 torch.no_grad() 와 다르다.

### torch.no_grad()

- PyTorch의 Autograd Engine을 비활성화 하여 Gradient 계산 않하도록 한다.
- Gradient 계산이 필요 없는 부분에 torch.no_grad() 사용하여 메모리 사용량 줄이고 계산 속도 향상

```python
model.eval()
with torch.no_graD():
```