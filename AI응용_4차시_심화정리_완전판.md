# AI응용 4차시 심화 정리본

## 신경망, 역전파, 최적화, 평가 지표 완전 정리

1.  이 차시의 본질

이번 차시는 딥러닝이 어떻게 학습되는지를 구조적으로 이해하는 차시다.
핵심 질문은 모델은 어떻게 틀린 만큼을 알고, 그걸 어떻게 고치는가이다.

2.  머신러닝 문제 정의

입력 X, 출력 Y, 과업(Task)을 먼저 정의한다. 문제 정의는 손실 함수와 평가
지표까지 결정한다.

3.  선형 모델

y = Wx + b 모든 신경망은 이 구조의 확장이다.

4.  학습 사이클

Forward → Loss → Backward → Update

5.  손실 함수

MSE = ((Y_pred - Y_true)\^2).mean()

6.  경사하강법

W = W - lr \* dL/dW

7.  역전파

연쇄법칙을 이용해 기울기를 계산한다. loss.backward() 한 줄로 수행된다.

8.  활성화 함수

ReLU가 현대 딥러닝의 기본 선택이다.

9.  Optimizer

optimizer.step() optimizer.zero_grad()

10. Bias-Variance

Underfitting vs Overfitting

11. 데이터 분할

Train / Validation / Test 분리

12. 평가 지표

회귀: MSE, MAE, R² 분류: Accuracy, Precision, Recall, F1, ROC-AUC
Confusion Matrix 기반 계산

13. 핵심 코드

``` python
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

def pred(X):
    return W * X + B

def mse(Yp, Y):
    return ((Yp - Y) ** 2).mean()

loss = mse(pred(X), Y)
loss.backward()
```
