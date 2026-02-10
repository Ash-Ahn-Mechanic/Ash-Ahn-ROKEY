# AI개론 3차시 — 핵심 코드 (Tensor · Autograd · Loss)

> 출력/그래프/폰트/시각화 코드는 제외하고 **개념을 대표하는 코드만** 남겼습니다.

---

## 1) 필수 임포트

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## 2) Tensor 생성 (0~3계) + NumPy → Tensor

```python
# 0계(스칼라)
r0 = torch.tensor(1.0, dtype=torch.float32)

# 1계(벡터): NumPy -> Tensor
r1_np = np.array([1, 2, 3, 4, 5])
r1 = torch.tensor(r1_np, dtype=torch.float32)

# 2계(행렬): NumPy -> Tensor
r2_np = np.array([[1, 5, 6],
                  [4, 3, 2]])
r2 = torch.tensor(r2_np, dtype=torch.float32)

# 3계(예: 채널 3개짜리 2x2 텐서)
torch.manual_seed(123)
r3 = torch.randn((3, 2, 2))
```

---

## 3) Tensor shape 변환 (view)

```python
# (3, 2, 2) -> (3, 4)
r3_2d = r3.view(3, -1)

# (3, 2, 2) -> (12,)
r3_1d = r3.view(-1)
```

---

## 4) Autograd: 2차 함수의 경사 계산 (requires_grad, sum, backward, grad)

```python
x_np = np.arange(-2, 2.1, 0.25)

# ✅ 미분 대상: requires_grad=True
x = torch.tensor(x_np, requires_grad=True, dtype=torch.float32)

# y = 2x^2 + 2
y = 2 * x**2 + 2

# ✅ backward()를 위해 스칼라로 만들기
z = y.sum()

# ✅ 경사 계산
z.backward()

# x.grad == dy/dx == 4x
grad = x.grad
```

---

## 5) Autograd 누적 방지: grad 초기화

```python
x.grad.zero_()  # 다음 backward 전에 반드시 초기화
```

---

## 6) Sigmoid의 경사 계산 (내장 연산도 자동 미분됨)

```python
sigmoid = nn.Sigmoid()

y = sigmoid(x)
z = y.sum()
z.backward()

sig_grad = x.grad
```

---

## 7) Loss Function: MSE (회귀)

```python
y_true = torch.tensor([3.0, 5.0, 2.0])
y_pred = torch.tensor([2.5, 4.5, 2.0])

mse_loss = nn.MSELoss()
loss = mse_loss(y_pred, y_true)
```

### (선택) MSE 직접 구현

```python
def mean_squared_error(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
```

---

## 8) Loss Function: Cross Entropy (분류)

```python
# (배치=5, 클래스=3) 로짓(logits)
logits = torch.randn(5, 3)

# 정답 레이블 (정수)
targets = torch.randint(0, 3, (5,))

ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, targets)
```


