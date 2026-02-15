# 📘 AI개론 4차시 정리본
**주제:** 인공지능 기초 — 신경망, 역전파(Backprop), 활성화 함수, 최적화, 데이터 분할/누수, 성능평가  
**기준:** 4차시 PDF 전 범위 + 4차시 ipynb(01/02/03) 핵심 코드만 선별하여 “왜/어떻게/어디에 쓰는지”까지 설명

---

## 목차
1. 수업 목표와 전체 큰그림  
2. AI/ML/DL 관계와 “왜 딥러닝인가”  
3. 학습 패러다임(지도/비지도/자기지도/강화) — 이 차시의 위치 고정  
4. 모든 딥러닝의 시작: 선형 모델(Linear Model)  
5. 문제 정의: X, Y, Task를 먼저 고정해야 하는 이유  
6. 학습 전체 사이클(Forward → Loss → Backward → Update)  
7. 경사하강법(Gradient Descent): 산 내려오기 비유를 “수식/코드”로 바꾸기  
8. 역전파(Backpropagation): 연쇄법칙과 Autograd  
9. 활성화 함수(Activation): 없으면 Deep이 불가능한 이유 + 대표 함수 비교  
10. 학습이 잘 안 되는 이유: 국소 최적해(Local Minimum)  
11. 왜 최적화(Optimizer)가 필요한가: SGD, Momentum, step/zero_grad  
12. 모델 용량(Capacity)과 Bias–Variance Trade-off  
13. 데이터 분할(Train/Val/Test)과 데이터 누수(Data Leakage) 방지  
14. K-Fold 교차검증  
15. 성능 평가 지표: 회귀/분류 + Confusion Matrix + ROC-AUC  
16. (부록) 이 차시를 “다음 차시(MLP/실전 학습)”로 연결하는 체크리스트

---

# 1) 수업 목표와 전체 큰그림

## 1.1 수업 목표(이 차시가 끝나면 할 수 있어야 하는 것)
- **문제 정의**: 입력(X)과 정답(Y) 설계, 어떤 Task인지(회귀/분류) 결정
- **경사 하강법** 이해: “손실을 줄이는 방향”으로 파라미터를 조금씩 수정
- **학습 루프 구현**: 전처리 → 예측 → 손실 → 기울기 → 업데이트 → 반복 → 평가
- **역전파**의 의미 이해: “오차를 뒤로 보내 각 파라미터의 책임(기여도)을 계산”
- **Optimizer**(SGD/momentum)의 의미 이해: 수동 업데이트를 더 안정적·빠르게
- **데이터 분할/누수 방지**: 평가 데이터를 학습에 섞으면 결과가 거짓말이 됨
- **평가 지표**(회귀/분류) 이해: 특히 분류는 Confusion Matrix 기반

## 1.2 전체 흐름(한 장 그림으로 끝내기)
아래 그림을 머릿속에 고정하면 4차시 전체가 한 줄로 연결된다.

```
[문제 정의] → [데이터 준비/전처리] → [모델(Linear)] → [예측(Forward)]
     → [손실(Loss)] → [역전파(Backward)] → [업데이트(Update)]
     → (반복) → [검증/평가] → [튜닝/개선]
```

---

# 2) AI / ML / DL 관계와 “왜 딥러닝인가”

## 2.1 관계(포함 관계)
- **AI ⊃ ML ⊃ DL**
  - AI: 사람처럼 판단/행동하는 기술 전체
  - ML: 데이터로부터 규칙을 “학습”하는 AI
  - DL: “여러 층(Deep)”을 쌓아 복잡한 패턴을 학습하는 ML

### 시각 자료(개념 맵)
```
AI  ────────────────────────────────┐
  ML ────────────────────────────┐  │
     DL ───────────────────────┐  │  │
                               └──┴──┘
```

## 2.2 이 차시에서 제일 중요한 문장
> “CNN, Transformer도 결국 **선형 변환(Linear) + 비선형(Activation)** 블록을 반복해서 쌓은 것이다.”

이 문장이 “왜 선형 모델부터 배우는가?”를 정당화한다.

---

# 3) 학습 패러다임(지도/비지도/자기지도/강화) — 이 차시의 위치

이 차시는 **지도학습(supervised)**을 기준으로 설명한다.  
왜냐하면 “입력-정답 쌍(X, Y)”을 두고 오차(Loss)를 줄이는 학습 구조가 가장 직관적이기 때문이다.

## 3.1 네 가지 패러다임 비교(암기표)
| 패러다임 | 정답(라벨) | 목표 | 예시 |
|---|---:|---|---|
| 지도학습 | 있음 | 분류/회귀 | 스팸 분류, 주택가격 예측 |
| 비지도학습 | 없음 | 패턴 발견 | 군집화, 차원축소 |
| 자기지도학습 | 없음(데이터로 생성) | 표현 학습 | Masked, Contrastive |
| 강화학습 | 보상만 | 행동 최적화 | 로봇, 게임 에이전트 |

---

# 4) 모든 딥러닝의 시작: 선형 모델(Linear Model)

## 4.1 선형 모델이 “모든 것의 씨앗”인 이유
딥러닝의 가장 기본 블록은 아래 식이다.

$$
\hat{y} = Wx + b
$$

- $$(x)$$: 입력
- $$(W)$$: 가중치(학습되는 값)
- $$(b)$$: 편향(학습되는 값)
- $$(\hat{y})$$: 예측값

## 4.2 핵심 사실: 선형 함수는 쌓아도 선형이다
**Linear + Linear = Linear**  
즉, 선형층만 여러 개 쌓아도 “결국 하나의 선형 모델”과 같아진다.

### 시각 자료(왜 깊이가 의미 없어지는가)
```
x ──[Linear]──> h ──[Linear]──> y
         (선형)        (선형)

두 개를 합치면:
x ───────────────[하나의 선형]──────────────> y
```

그래서 “Deep”이 되려면 반드시 **활성화 함수(비선형)**가 들어가야 한다.

---

# 5) 문제 정의: X, Y, Task를 먼저 고정해야 하는 이유

## 5.1 문제 정의가 바뀌면, 모든 것이 바뀐다
- Task가 회귀면: 손실(MSE/MAE), 지표(R² 등), 출력(실수)
- Task가 분류면: 손실(CrossEntropy/BCE), 지표(Accuracy/F1/ROC-AUC), 출력(클래스/확률)

## 5.2 4차시의 예시 문제(회귀)
> “신장(x)으로 체중(y)을 예측한다.”

이때:
- 입력 $$(X)$$: 신장(cm)
- 정답 $$(Y)$$: 체중(kg)
- Task: 회귀(Regression)

---

# 6) 학습 전체 사이클(Forward → Loss → Backward → Update)

이 네 단계는 **딥러닝의 표준 문장**이다.

1) **Forward(예측)**: $$( \hat{y} = f(x) )$$  
2) **Loss(손실)**: $$( L(\hat{y}, y) )$$  
3) **Backward(역전파)**: $$( \nabla_W L, \nabla_b L )$$  
4) **Update(업데이트)**: $$( W \leftarrow W - lr \cdot \nabla_W L )$$

### 시각 자료(한 줄로 고정)
```
X ──(Forward)──> Y_hat ──(Loss)──> L
                        ▲          │
                        │          │
                 (Update)      (Backward)
                        │          │
                        └── W,b ───┘
```

---

# 7) 경사하강법(Gradient Descent): “산 내려오기”를 수식/코드로

## 7.1 산 내려오기 비유를 학습 루프에 매핑
- 현재 고도 = 손실(Loss)
- 발의 기울기 = 그래디언트(Gradient)
- 보폭 = 학습률(Learning Rate)

### “비유 → 수식”
$$
W \leftarrow W - lr \cdot \frac{\partial L}{\partial W}
$$

- \(lr\)이 너무 크면: **오버슈팅(발산)**
- \(lr\)이 너무 작으면: **너무 느림(수렴 지연)**

---

# 8) 역전파(Backpropagation): 연쇄법칙 + Autograd

## 8.1 역전파의 핵심 정의
> 출력의 오차를 입력 방향으로 되돌리며,  
> 각 파라미터(W, b)가 오차에 얼마나 책임이 있는지(기여도)를 계산한다.

## 8.2 왜 “뒤에서 앞으로”인가?
- Loss는 단 하나의 값
- 원인(파라미터)은 수만 개  
→ 결과에서 시작해서 원인으로 거슬러 올라가야 “책임 분배”가 가능

## 8.3 연쇄 법칙(Chain Rule) 한 번에 이해하기
아래처럼 연결된 함수가 있다고 하자.

$$[
L = L(\hat{y}), \quad \hat{y} = f(W)
]$$

텍스트 표현:

    dL/dW = (dL/dy_hat) * (dy_hat/dW)

LaTeX 표현 (수식 지원 환경):

$$
\begin{aligned}
\frac{\partial L}{\partial W}
&=
\frac{\partial L}{\partial \hat{y}}
\cdot
\frac{\partial \hat{y}}{\partial W}
\end{aligned}
$$


즉, **복잡한 미분을 조각으로 나누고 곱해서 연결**한다.

## 8.4 PyTorch에서는 어떻게 되나?
- `requires_grad=True`를 켜면
- PyTorch가 계산 그래프를 저장한다
- `loss.backward()`를 하면 그래프를 거꾸로 돌면서 기울기를 누적한다
- 결과는 각 파라미터의 `.grad`에 저장된다

---

# 9) 활성화 함수(Activation): 없으면 Deep이 불가능한 이유

## 9.1 활성화 함수가 없으면?
앞서 말한 대로:

- 선형의 합성은 선형
- 그러면 층을 아무리 깊게 쌓아도 결국 “하나의 선형 회귀” 수준

결론:
> 활성화 함수가 없으면 “Deep Learning”이 성립하지 않는다.

## 9.2 대표 활성화 함수 비교(시험 포인트)
| 함수 | 출력 범위 | 장점 | 치명적 단점 |
|---|---:|---|---|
| Sigmoid | 0~1 | 확률 해석 | 양 끝 포화 → 기울기 소실 |
| Tanh | -1~1 | zero-centered | 포화 영역 기울기 소실 |
| ReLU | 0~∞ | 빠름/희소성 | 음수 영역 죽음(Dying ReLU) |

---

# 10) 학습이 잘 안 되는 이유: 국소 최적해(Local Minimum)

딥러닝의 손실 지형은 보통 “울퉁불퉁(비볼록)”이다.  
그래서 단순 GD는 **국소 최적해**에 갇힐 수 있다.

### 핵심 문장
> “어디서 출발하느냐(초기값)가 도착지를 결정할 수 있다.”

---

# 11) 왜 최적화(Optimizer)가 필요한가: SGD, Momentum, step/zero_grad

## 11.1 수동 업데이트의 한계
수동으로는 이런 걸 직접 처리해야 한다:
- 업데이트 수식 적용
- grad 초기화
- 모멘텀 같은 개선 기법 구현

## 11.2 Optimizer는 “전문 코치”
- `optimizer.step()` 한 줄로 파라미터 업데이트
- `optimizer.zero_grad()`로 grad 초기화

## 11.3 Momentum이 왜 빠른가(직관)
- 이전 이동 방향(관성)을 기억해서
- 진동을 줄이고, 골짜기 방향으로 더 빨리 내려가게 만든다

---

# 12) 모델 용량(Capacity)과 Bias–Variance Trade-off

## 12.1 모델 용량이란?
모델이 표현할 수 있는 복잡함의 정도(공부 능력).

- 낮은 용량: 단순 패턴만 가능 → **과소적합(Underfitting)**
- 높은 용량: 복잡한 패턴 가능(하지만 노이즈도 외움) → **과적합(Overfitting)**

## 12.2 Bias vs Variance(암기 포인트)
- Underfitting: Train/Val 둘 다 성능 낮음 → **High Bias**
- Overfitting: Train 좋고 Val 나쁨 → **High Variance**

---

# 13) 데이터 분할(Train/Val/Test)과 데이터 누수(Data Leakage) 방지

## 13.1 왜 나누나? (시험 비유)
- Train: 교과서로 공부
- Val: 모의고사로 튜닝
- Test: 실전 수능(최종 평가)

## 13.2 데이터 누수란?
Test/Val 정보를 학습에 섞어버리는 것.  
그럼 성능이 **좋아 보이기만** 하고 실제 성능이 아니다.

---

# 14) K-Fold 교차검증

데이터가 적을 때 유리:
- 데이터를 K개로 나누고
- 돌아가며 검증 세트로 사용
- K번 성능 평균이 최종 성능

---

# 15) 성능 평가 지표: 회귀/분류 + Confusion Matrix + ROC-AUC

## 15.1 회귀(Regression) 지표
- MSE: 큰 오차에 민감
- MAE: 오차를 직관적으로 해석(절대값)
- R²: 설명력(0~1 근처로 갈수록 좋음)

## 15.2 분류(Classification) 지표
### Confusion Matrix(혼동행렬)
|  | Pred + | Pred - |
|---|---:|---:|
| True + | TP | FN |
| True - | FP | TN |

- **Accuracy**: (TP+TN)/전체  
  - 불균형 데이터에서 함정(암 환자 1%면 전부 정상 예측해도 99% 나옴)
- **Precision**: TP/(TP+FP)  
  - “양성이라고 말한 것 중 진짜 양성 비율” (스팸 필터)
- **Recall**: TP/(TP+FN)  
  - “진짜 양성 중 찾아낸 비율” (암 진단)
- **F1**: Precision과 Recall의 조화평균  
  - 불균형 데이터 기본 추천

### ROC-AUC
- 임계값(threshold)을 바꿔가며
- TPR(Recall=TP/(TP+FN)) vs FPR(FP/(FP+TN))을 그린 ROC curve 아래 면적
- 0.5: 랜덤 / 1.0: 완벽

---

# 16) 코드-이론 1:1 매핑 (ipynb 핵심 코드만)

아래 코드는 “이 차시 이론을 실제로 구현한 최소 단위”만 남겼다.  
각 블록마다:
- **PDF의 어떤 이론을 구현하는지**
- **이 코드에서 절대 놓치면 안 되는 핵심**
- **왜 효용이 있는지(실전/시험/다음 차시 연결)**  
을 함께 적는다.

---

## 16.1 (이론: 데이터 준비/전처리) 평균 중심화로 학습 안정화
**PDF 파트:** 데이터 준비 및 전처리 / 데이터 변환(평균값 빼기)

```python
import numpy as np
import torch

sampleData1 = np.array([
    [166, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
])

# X(신장), Y(체중) 분리
x = sampleData1[:, 0]
y = sampleData1[:, 1]

# 평균 중심화: 값의 중심을 0으로 (경사하강 안정화에 도움)
X = x - x.mean()
Y = y - y.mean()

# torch 텐서로 변환 (autograd 사용 기반)
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
```

### ✅ 핵심 강조
- `X = x - x.mean()`가 중요한 이유  
  → 값 스케일이 줄고 중심이 0으로 가서 **학습이 더 안정**해질 가능성이 커진다.
- 이건 “정답을 바꾸는” 게 아니라 “좌표계를 옮기는” 것(관계는 유지)

### 🔜 다음 차시 확장
- 표준화(Standardization), 정규화(Normalization)
- 실제 데이터셋에서 `Dataset / DataLoader`

---

## 16.2 (이론: 선형모델) y = Wx + b를 그대로 코드로
**PDF 파트:** 선형 모델 / 왜 선형회귀로부터 배우는가

```python
# 학습될 파라미터(가중치/편향)
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# 예측 함수: y_hat = W*X + B
def pred(X):
    return W * X + B
```

### ✅ 핵심 강조
- `requires_grad=True`  
  → 이 스위치가 꺼지면 `loss.backward()` 해도 **W,B의 기울기를 계산할 수 없다.**
- 딥러닝의 모든 레이어는 이 구조의 확장 (`nn.Linear`로 일반화)

### 🔜 다음 차시 확장
- `nn.Linear(in_features, out_features)`
- 다층 퍼셉트론: Linear + Activation 반복

---

## 16.3 (이론: 손실 함수) MSE 구현
**PDF 파트:** 손실 계산 / 회귀 지표(MSE)

```python
def mse(Yp, Y):
    loss = ((Yp - Y)**2).mean()
    return loss
```

### ✅ 핵심 강조
- 손실 함수는 “학습 목표” 자체다.  
  무엇을 최소화할지 정하지 않으면 **학습이 성립하지 않는다.**

---

## 16.4 (이론: 역전파) loss.backward()가 하는 일
**PDF 파트:** 역전파 정의 / 연쇄법칙 / PyTorch에서의 역전파

```python
Yp = pred(X)
loss = mse(Yp, Y)

loss.backward()  # 역전파: W.grad, B.grad가 채워짐
```

### ✅ 핵심 강조
- backward 이후:
  - `W.grad`: 손실을 줄이려면 W를 어느 방향으로 바꿔야 하는지
  - `B.grad`: 동일
- 이게 곧 “책임 분배”의 수치화 결과

---

## 16.5 (이론: 업데이트) in-place 에러와 torch.no_grad()
**PDF 파트:** 파라미터 수정(올바른 방법)

```python
lr = 0.001

with torch.no_grad():
    W -= lr * W.grad
    B -= lr * B.grad

W.grad.zero_()
B.grad.zero_()
```

### ✅ 핵심 강조 1 — 왜 `torch.no_grad()`가 필요한가
- `W -= ...`는 **in-place 연산**
- autograd가 추적하는 leaf tensor(W,B)에 in-place로 손대면 그래프가 깨질 수 있어 PyTorch가 막는다.
- `no_grad()`는 “이 업데이트는 미분 추적하지 마”라는 의미

### ✅ 핵심 강조 2 — 왜 `zero_()`가 필요한가
- PyTorch는 기본적으로 `.grad`를 **누적(accumulate)**한다.
- `zero_grad`를 안 하면  
  이번 epoch의 기울기 + 다음 epoch 기울기가 계속 쌓여서 학습이 망가진다.

---

## 16.6 (이론: 학습 루프) 학습 사이클 4단계를 코드로 고정
**PDF 파트:** 학습 전체 사이클 / 경사하강법 4단계

```python
num_epochs = 500
hist = np.zeros((0, 2))

for epoch in range(num_epochs):
    # 1) Forward
    Yp = pred(X)

    # 2) Loss
    loss = mse(Yp, Y)

    # 3) Backward
    loss.backward()

    # 4) Update
    with torch.no_grad():
        W -= lr * W.grad
        B -= lr * B.grad

    # grad reset
    W.grad.zero_()
    B.grad.zero_()

    # 기록(학습곡선)
    if (epoch % 10 == 0):
        item = np.array([epoch, loss.item()])
        hist = np.vstack((hist, item))
```

### ✅ 핵심 강조
- 이 루프 자체가 딥러닝의 표준 문장이다.  
  나중에 CNN이든 Transformer든 **뼈대는 그대로**다.

---

## 16.7 (이론: Optimizer) step / zero_grad로 실전 형태 완성
**PDF 파트:** 왜 최적화가 필요한가 / 최적화 함수 구현

```python
import torch.optim as optim

optimizer = optim.SGD([W, B], lr=lr)

for epoch in range(num_epochs):
    Yp = pred(X)
    loss = mse(Yp, Y)

    loss.backward()

    optimizer.step()      # Update
    optimizer.zero_grad() # grad reset
```

### ✅ 핵심 강조
- `optimizer.step()` = “W,B 업데이트를 표준 방식으로 수행”
- `optimizer.zero_grad()` = “누적 방지”
- 실전 코드에서는 대부분 이 패턴을 쓴다.

---

## 16.8 (이론: 튜닝) Momentum으로 수렴 가속
**PDF 파트:** 최적화 함수 튜닝(momemtum)

```python
optimizer = optim.SGD([W, B], lr=lr, momentum=0.9)
```

### ✅ 핵심 강조
- momentum은 “관성”  
  → 진동을 줄이고 빠르게 내려가게 도와준다.

---

## 16.9 (이론: Bias–Variance) 저용량 vs 고용량 MLP로 체감
**PDF 파트:** 모델 용량 / Bias–Variance Trade-off / 데이터 분할 이유

```python
import torch, math
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 데이터: sin + noise
torch.manual_seed(0)
N = 600
x = torch.linspace(-3*math.pi, 3*math.pi, N).unsqueeze(1)
y = torch.sin(x) + 0.2*torch.randn_like(x)

# Train/Val/Test 분할
X_train, X_temp, y_train, y_temp = train_test_split(x.numpy(), y.numpy(), test_size=0.4, random_state=42)
X_val,   X_test, y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val,   y_val   = torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.float32)
X_test,  y_test  = torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.float32)

# 모델 용량 비교
def make_mlp(hidden):
    return nn.Sequential(
        nn.Linear(1, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1)
    )

small = make_mlp(hidden=8)    # 고편향/저분산 경향
big   = make_mlp(hidden=128)  # 저편향/고분산 경향
```

### ✅ 핵심 강조
- “모델 용량”은 그냥 말로만 이해하면 헷갈린다.  
  **작은 모델 vs 큰 모델을 같은 데이터로 학습**하면  
  Train/Val 성능 격차로 **과적합/과소적합 감각이 생긴다.**

---

# 17) 시험/과제용 핵심 체크리스트(이 차시 끝나면 바로 답할 수 있어야 함)

✅ **(1)** Forward/Loss/Backward/Update를 말로 설명하고 코드로도 쓸 수 있는가?  
✅ **(2)** `requires_grad=True`의 역할을 정확히 말할 수 있는가?  
✅ **(3)** `loss.backward()`가 무엇을 채우는지(W.grad/B.grad)를 말할 수 있는가?  
✅ **(4)** `torch.no_grad()`가 필요한 이유(in-place + autograd) 설명 가능한가?  
✅ **(5)** `zero_grad()`를 안 하면 왜 망하는지(누적) 설명 가능한가?  
✅ **(6)** Underfitting/Overfitting을 Train/Val 그래프로 구분할 수 있는가?  
✅ **(7)** 데이터 누수(Data Leakage) 정의 + 왜 위험한지 예시로 말할 수 있는가?  
✅ **(8)** Confusion Matrix에서 Precision/Recall/F1 유도할 수 있는가?  
✅ **(9)** ROC-AUC가 “threshold 변화에 대한 종합 지표”임을 설명할 수 있는가?

---

# 18) 다음 차시로 연결(이 차시를 발판으로 무엇을 더 배우나)

- 선형 모델 \(y=Wx+b\) → `nn.Linear` 일반화 → **MLP 구조**
- 수동 GD → Optimizer/스케줄러/정규화 → **실전 학습 파이프라인**
- 회귀(MSE) → 분류(CrossEntropy/BCE) → **Confusion Matrix/ROC-AUC 실습**
- 단일 변수(1D) → 다변수(벡터/행렬) → **진짜 “신경망” 형태**

---

> ✅ 끝.  
> 이 문서는 “4차시 PDF 내용 전체를 빠짐없이 한 흐름으로 묶고”,  
> “ipynb 코드가 PDF 이론의 어느 지점을 구현하는지”를 1:1로 맞춰 만든 교재형 정리본이다.
