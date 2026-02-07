# 2차시(OpenCV 기하학적 변환) — 과정 3: 차시 핵심 코드 섹션

> 목적: 2차시에서 **반드시 가져가야 할 핵심 코드 패턴(🔥)**만 남긴 요약본  
> 핵심은 “**좌표를 바꾸는 행렬을 만들고 → warp로 매핑한다**” 입니다.

---

## 1) 크기 변환 + 보간(Interpolation) — `cv2.resize`

확대/축소는 픽셀을 새 격자에 “재배치”하는 과정이라 **빈 픽셀을 채우는 보간**이 필수입니다.

```python
import cv2

src = cv2.imread(image_path)

# nearest: 가장 가까운 픽셀을 그대로 복사(빠름, 계단 현상 가능)
dst_nearest = cv2.resize(src, dsize=(560, 560), interpolation=cv2.INTER_NEAREST)

# cubic: 주변을 더 많이 참고해 부드럽게(선명/부드러움, 계산량 증가)
dst_cubic = cv2.resize(src, dsize=(720, 720), interpolation=cv2.INTER_CUBIC)
```

**왜 핵심?**
- 데이터 증강(augmentation)에서 “스케일 변화”는 기본 중 기본
- 객체 크기 보정, 입력 해상도 통일 파이프라인에 그대로 사용

---

## 2) 회전(Rotation) — `getRotationMatrix2D` + `warpAffine`

“회전 행렬(2×3)을 만든 뒤, `warpAffine`으로 픽셀 좌표를 새로 찍는다”가 핵심입니다.

```python
import cv2

src = cv2.imread(image_path)
h, w = src.shape[:2]
center = (w / 2, h / 2)

angle = 90     # (+)면 반시계
scale = 0.5

M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(src, M, (w, h))
```

**왜 핵심?**
- 기하학 변환의 본질(행렬 생성 → 좌표 매핑)을 가장 직관적으로 보여줌
- 회전/이동/스케일은 Affine의 부분집합(다음 Affine 학습 연결)

---

## 3) Affine(평행이동/기울임/스케일 포함) — `warpAffine` (2×3 행렬)

Affine은 “선형변환 + 평행이동”이고, 이를 **2×3 행렬**로 표현합니다.

### 3-1) 평행이동(Translation) 예시
```python
import cv2
import numpy as np

src = cv2.imread(image_path)
h, w = src.shape[:2]

tx, ty = 50, 30  # 오른쪽 50px, 아래 30px 이동
A = np.array([[1.0, 0.0, tx],
              [0.0, 1.0, ty]], dtype=np.float32)

shifted = cv2.warpAffine(src, A, (w, h))
```

**왜 핵심?**
- 2×3 행렬이 “좌표를 어떻게 바꾸는지”를 직접 보여줌
- 이후 Shear/Rotation/Scale을 같은 틀(행렬)로 묶어서 이해 가능

---

## 4) Affine(3점 매핑) — `getAffineTransform` + `warpAffine`

Affine은 **3개의 대응 점(삼각형)**만 정하면 변환이 결정됩니다.

```python
import cv2
import numpy as np

img = cv2.imread(image_path)
h, w = img.shape[:2]

src_pts = np.float32([[50, 50], [350, 50], [50, 350]])
dst_pts = np.float32([[80, 100], [320, 80], [100, 320]])

M = cv2.getAffineTransform(src_pts, dst_pts)
affine_img = cv2.warpAffine(img, M, (w, h))
```

**왜 핵심?**
- “점 3개로 행렬이 정해진다” = **Affine의 핵심 메커니즘(6 DOF)**
- 기울어진 이미지 정렬/보정, 전처리 정합(alignment)로 바로 확장

---

## 5) Perspective(투시/원근) 변환 — `getPerspectiveTransform` + `warpPerspective`

원근 효과는 평행성이 깨지기 때문에 Affine으로는 불가하고, **4점 대응**이 필요합니다.

```python
import cv2
import numpy as np

img = cv2.imread(image_path)
h, w = img.shape[:2]

# 원본 4점(사각형)
src_pts = np.float32([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
])

# 목표 4점(사다리꼴) — 원근감 생성/보정
dst_pts = np.float32([
    [50, 100],
    [w - 50, 100],
    [w - 20, h - 50],
    [20, h - 50]
])

H = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, H, (w, h), borderValue=(200, 200, 200))
```

**왜 핵심?**
- 문서 스캐너(비스듬한 문서 → 정면 보정), 도로/건물 왜곡 보정의 기본
- “4점 → 3×3 행렬 → warpPerspective” 패턴이 실무에서 그대로 반복됨

---

## ⚠️ (실무 체크) OpenCV 색상(BGR) vs Matplotlib(RGB)
- `cv2.imread()`는 **BGR**
- `matplotlib.pyplot.imshow()`는 **RGB**로 보는 경우가 많아서 색이 이상해 보일 수 있음  
→ 출력할 때만 변환 추천:

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
```

---

## 요약: 2차시 핵심 패턴 1줄
- **행렬 만들기(getRotationMatrix2D / getAffineTransform / getPerspectiveTransform) → 워핑(warpAffine / warpPerspective)**  
이 흐름만 확실히 잡으면 다음 차시(응용/검출 전처리)로 자연스럽게 이어집니다.
