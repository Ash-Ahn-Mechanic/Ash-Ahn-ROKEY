# AI응용 4차시 핵심 이론 및 핵심 코드

## 1. 차시 핵심 이론 요약

### AI / ML / DL 관계

AI ⊃ ML ⊃ DL\
딥러닝은 선형 변환과 비선형 활성화 함수의 반복 구조를 기반으로 한다.

### 머신러닝 문제 정의

입력(X), 출력(Y), 과업(Task)을 먼저 정의한다. 문제 정의는 데이터 구조,
손실 함수, 평가 지표를 결정한다.

### 선형 모델

y = Wx + b\
모든 신경망의 최소 단위.

### 학습 전체 사이클

Forward → Loss → Backward → Update

### 역전파

출력의 오차를 입력 방향으로 전달하며 각 파라미터의 기여도를 계산한다.

### 활성화 함수

선형 결합만으로는 표현력 한계가 존재. ReLU, Sigmoid, Tanh 등이 사용된다.

### 최적화

SGD, Momentum, Adam 등으로 학습을 가속화한다.

### Bias--Variance Trade-off

Underfitting vs Overfitting, Sweet Spot 찾기.

### 데이터 분할

Train / Validation / Test 분리 데이터 누수 방지 필수.

### 성능 평가 지표

-   회귀: MSE, MAE, R²
-   분류: Accuracy, Precision, Recall, F1, ROC-AUC
-   Confusion Matrix

------------------------------------------------------------------------

## 2. 차시 핵심 코드

### 데이터 준비 및 전처리

``` python
sampleData1 = np.array([
    [166, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
])

x = sampleData1[:, 0]
y = sampleData1[:, 1]

X = x - x.mean()
Y = y - y.mean()

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
```

### 선형 모델 정의

``` python
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

def pred(X):
    return W * X + B
```

### 손실 함수 (MSE)

``` python
def mse(Yp, Y):
    return ((Yp - Y) ** 2).mean()
```

### 역전파

``` python
loss.backward()
```

### 경사하강법

``` python
with torch.no_grad():
    W -= lr * W.grad
    B -= lr * B.grad
```

### Optimizer

``` python
optimizer = torch.optim.SGD([W, B], lr=lr)
optimizer.step()
optimizer.zero_grad()
```

------------------------------------------------------------------------

## 3. 핵심 요약

-   딥러닝 학습은 4단계 사이클
-   역전파는 핵심 메커니즘
-   데이터 누수는 치명적 오류
-   분류 문제는 Confusion Matrix 기반 평가
