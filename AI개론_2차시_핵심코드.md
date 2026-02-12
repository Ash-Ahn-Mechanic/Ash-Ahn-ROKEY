# AI개론_2차시_핵심코드.md
> 7기 AI개론 2차시 (NumPy / Matplotlib) — **핵심 코드만**  
> 기준: 실무·시험·다음 차시 연결

---

## 0) 공통 임포트
```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## 1) ndarray 생성 + Shape/Dtype 확인
```python
a = np.array([1, 2, 3], dtype=np.float32)

print(a.shape)  # (3,)
print(a.dtype)  # float32
```
- 왜 핵심: **Shape/Dtype**은 텐서 차원/연산의 기본 뼈대

---

## 2) 벡터화(Vectorization) — for문 제거
```python
data = np.array([1, 2, 3])

out = data * 2   # for문 없이 전체가 한 번에 계산
print(out)
```
- 왜 핵심: 딥러닝 연산(대량 데이터)이 성립하는 핵심 습관 = **배열 전체 연산**

---

## 3) Broadcasting — Batch에 Bias 한 줄로 더하기
```python
batch = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])     # (3, 3)

bias = np.array([10, 10, 10])     # (3,)

y = batch + bias                   # (3, 3) + (3,) -> (3, 3)
print(y)
```
- 왜 핵심: **복사 없이(No Copy)** 형상을 맞춰 연산 → 배치 처리/편향(bias) 처리의 기본

---

## 4) Aggregation — Loss는 왜 숫자 1개(스칼라)인가
```python
pred = np.array([0.2, 0.7, 0.1])
target = np.array([0, 1, 0])

error = (pred - target) ** 2       # (3,)  아직 배열
loss = np.mean(error)              # 스칼라 1개

print(loss)
```
- 왜 핵심: Loss = (오차 배열) → **집계(mean/sum)** 로 스칼라 만들기

---

## 5) Axis — Batch vs Feature 방향 구분
```python
X = np.array([[10, 20, 30],
              [40, 50, 60]])       # (2 samples, 3 features)

mean_over_batch = np.mean(X, axis=0)   # feature별 평균 (3,)
mean_over_feat  = np.mean(X, axis=1)   # sample별 평균  (2,)

print(mean_over_batch)
print(mean_over_feat)
```
- 왜 핵심: axis=0(배치/샘플), axis=1(특징) 구분은 이후 (N,C,H,W) 차원 해석으로 직결

---


---

## 6) reshape / transpose — 차원(축) 조작의 기본
```python
# reshape: 원소 개수는 유지하고, 모양(shape)만 바꿈
x = np.arange(12)          # (12,)
x2 = x.reshape(3, 4)       # (3,4)

# -1: 남는 차원 자동 계산
x3 = x.reshape(2, -1)      # (2,6)

print(x2.shape, x3.shape)

# transpose / .T : 축 순서 바꾸기 (2D 전치)
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2,3)

AT = A.T                   # (3,2)
print(AT.shape)

# N차원 축 재배치
B = np.random.randn(2, 3, 4)        # (N, C, W) 같은 예시
B2 = np.transpose(B, (0, 2, 1))      # (2, 4, 3)
print(B2.shape)
```
- 왜 핵심: 딥러닝 데이터는 결국 **차원 싸움**이라서 (N,C,H,W) 정리/변환이 필수
- 실무 팁: `transpose` 후 `reshape`는 연속 메모리 문제가 생길 수 있어, PyTorch에선 보통 `reshape()`가 더 안전

## 7) Boolean Indexing — 조건 필터링(전처리)
```python
data = np.array([10, -5, 30, -2, 15])

filtered = data[data > 0]   # 조건 마스크로 필터링
print(filtered)
```
- 왜 핵심: if문 없이 **마스크 기반 전처리**(이상치 제거/조건 추출)

---

## 8) Matplotlib — 학습/데이터를 진단하는 최소선
```python
loss = [1.0, 0.8, 0.65, 0.55, 0.52]

plt.plot(loss, label="Loss")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()
```
- 왜 핵심: 시각화는 “그림”이 아니라 **진단(디버깅)** 도구 (Loss 곡선 해석의 기본)

---

