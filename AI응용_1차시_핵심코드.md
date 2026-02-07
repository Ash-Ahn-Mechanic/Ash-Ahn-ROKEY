# 8. 차시 핵심 코드 선별 및 정리

본 차시의 목적은 **디지털 영상의 개념 이해와 OpenCV 기반 기본 연산 흐름을 체득**하는 데 있으므로,  
실무·학습 관점에서 반드시 익혀야 할 코드만을 핵심으로 남기고 나머지는 탈락시킨다.

---

## 8.1 반드시 남겨야 할 핵심 코드

### (1) 영상 입출력 (모든 실습의 출발점)
```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')          # 영상 읽기 (BGR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')
```
**선정 이유**
- OpenCV 영상 처리의 시작점  
- BGR → RGB 변환 개념 필수  
- 이후 모든 연산의 입력 데이터 생성 단계

---

### (2) 색공간 변환: RGB → HSV
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```
**선정 이유**
- RGB vs HSV 개념을 코드로 직접 확인  
- 색상 기반 분할, 객체 검출의 핵심 전처리  
- 조명 변화에 강한 영상 처리 흐름 이해

---

### (3) 색상 검출 + 마스크 생성 (핵심 중 핵심)
```python
lower = (100, 100, 100)
upper = (140, 255, 255)

mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(img, img, mask=mask)
```
**선정 이유**
- 색공간 + 논리연산 + 마스크 개념 결합  
- 컴퓨터비전 실전에서 가장 많이 쓰이는 패턴  
- 객체 추적, ROI 추출의 최소 단위

---

### (4) 픽셀 산술 연산 – 밝기 조절
```python
bright = cv2.add(img, 50)
dark = cv2.subtract(img, 50)
```
**선정 이유**
- 포화 연산(saturation) 개념 체감  
- NumPy 연산과 OpenCV 연산 차이 이해  
- 영상 전처리의 기본기

---

### (5) 가중합 연산 (영상 블렌딩)
```python
blend = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
```
**선정 이유**
- 선형 결합 개념을 직관적으로 이해  
- 트랜지션, 합성, 데이터 증강과 연결

---

### (6) 절대값 차이 (모션 검출의 출발점)
```python
diff = cv2.absdiff(frame1, frame2)
```
**선정 이유**
- 시간 축 기반 영상 분석의 핵심  
- 모션 검출, CCTV, Optical Flow로 확장 가능

---

### (7) 히스토그램 평활화 (명암 개선의 핵심)
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)
```
**선정 이유**
- 히스토그램 기반 전처리 대표 사례  
- 대비 향상 효과를 즉시 확인 가능  
- CLAHE 등 고급 기법으로 확장 가능

---

## 8.2 탈락시킨 코드 유형과 이유

| 탈락 코드 | 탈락 이유 |
|---------|---------|
| 단순 출력용 코드 | 개념 이해 기여도 낮음 |
| 동일 기능의 중복 예제 | 학습 피로도 증가 |
| 수식 설명 없는 히스토그램 계산 | 이론 이해 없이 사용 위험 |
| 장식 목적 시각화 코드 | 차시 핵심과 무관 |

---

## 8.3 1차시 코드 학습 핵심 요약
- **영상은 행렬이다** → 모든 연산은 수치 연산  
- **색공간 변환은 필수**  
- **마스크 + bitwise = 컴퓨터비전 기본기**  
- **히스토그램은 영상 품질을 수치로 다루는 출발점**
