# AI응용_3차시_핵심코드.md
> 주제: OpenCV 기반 영상 특징 검출 (변화 → 엣지 → 형태/도형 → 객체)

---

## 0) 공통 입력 형태
```python
import cv2
import numpy as np

img = cv2.imread("your_image.png")              # BGR
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Gray
```

- **왜 핵심?** 모든 후속 특징 검출(이진화/엣지/윤곽/허프)은 `gray` 기준으로 진행되는 경우가 많음.
- **대표 개념:** 픽셀 기반 입력을 “처리 가능한 형태(Gray)”로 표준화.
- **다음 확장:** 영상(프레임) 단위 처리 시에도 동일하게 적용.

---

## 1) 🔥 이진화(Thresholding): 픽셀 → 영역(객체 후보)
```python
# (A) 전역 이진화: 하나의 임계값
_, binary_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# (B) 적응형 이진화: 조명 변화에 강함(영역마다 임계값 다름)
binary_ad = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 4
)

# (C) Otsu: 최적 임계값을 자동 선택(전역 자동화)
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

- **왜 핵심?** 엣지/윤곽/객체 검출은 “배경 vs 물체”가 갈라져야 시작됨(픽셀 → 구조).
- **대표 개념:** 변화(Feature)를 “분리된 영역”으로 만드는 첫 단계.
- **다음 확장:** morphology(열기/닫기)로 노이즈 정리 → `findContours` 안정화 / 객체 라벨링.

---

## 2) 🔥 Sobel + magnitude: 엣지 = 밝기 변화율(미분)
```python
# Sobel (미분) : 방향별 변화량
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)   # x방향 변화
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)   # y방향 변화

# 변화의 “세기”만 뽑기 (방향 무시)
mag = cv2.magnitude(gx, gy)
mag = np.uint8(np.clip(mag, 0, 255))
```

- **왜 핵심?** “특징=변화지점”을 수학(그래디언트/1차 미분)으로 구현하는 대표 코드.
- **대표 개념:** 차분(Differencing) → 엣지(Edge)로 연결.
- **다음 확장:** 방향(각도)까지 쓰면 코너/추적/옵티컬플로우 전처리로 확장 가능.

---

## 3) 🔥 Canny: 엣지 검출 파이프라인 표준
```python
edges = cv2.Canny(gray, 50, 150)   # (low, high) threshold
```

- **왜 핵심?** 내부적으로 노이즈 억제 + 얇은 엣지 + 강/약 엣지 연결성까지 포함한 “완성형 엣지”.
- **대표 개념:** 안정적인 엣지(One-pixel width) 생성.
- **다음 확장:** `Hough`, `Contours`, 객체 검출/추적 파이프라인의 표준 전처리.

---

## 4) 🔥 Contours: 중간수준 특징(형태)로 승격
```python
# 1) 이진화(대표로 Otsu 사용)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2) 윤곽선 추출
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# (다음 차시/응용 확장 예시)
# area = cv2.contourArea(contours[i])
# x,y,w,h = cv2.boundingRect(contours[i])
```

- **왜 핵심?** 픽셀/엣지를 넘어 “객체 후보의 외곽(형태)”를 직접 얻는 단계.
- **대표 개념:** 중간수준 특징(Mid-level feature) = 윤곽/형태 기술자의 시작.
- **다음 확장:** 면적/둘레/원형도/convex hull 등 shape descriptor, 객체 필터링/분할.

---

## 5) 🔥 Hough (직선/원): 엣지 → 도형(객체) 판정
### (A) 원 검출: `HoughCircles`
```python
blur = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=50
)

# circles: (1, N, 3) 형태 (x, y, r)
```

### (B) 직선(선분) 검출: `HoughLinesP`
```python
edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi/180,
    threshold=10,
    minLineLength=10,
    maxLineGap=10
)

# lines: (N, 1, 4) 형태 (x1, y1, x2, y2)
```

- **왜 핵심?** 엣지(점 집합)를 파라미터 공간에서 누적 투표로 “도형 존재”로 판정하는 전통 CV 객체 검출의 정점.
- **대표 개념:** 이미지 공간 → 파라미터 공간 변환 + 누적(Voting) + 강건성(노이즈/끊김).
- **다음 확장:** Hough 결과를 ROI로 사용 → 추적/인식(딥러닝) 파이프라인과 결합.

---
