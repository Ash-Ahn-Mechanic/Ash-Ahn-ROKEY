# AI응용_4차시_핵심코드.md
> 주제: 추적 알고리즘(Tracking) — 템플릿/색상분포/Optical Flow

---

## 0) 공통: 비디오 프레임 읽기(최소 형태)
```python
import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()
if not ret:
    raise RuntimeError("첫 프레임을 읽지 못했습니다.")
```

- **왜 핵심?** Tracking은 “연속 프레임”이 전제이므로, 최소한의 프레임 입력 구조는 반드시 필요.
- **대표 개념:** 시간축 입력(프레임 스트림).
- **다음 확장:** 파일/웹캠/실시간 스트림 모두 동일 구조로 확장.

---

## 1) 🔥 Template Matching: “왜 더 좋은 추적이 필요한가” 기준점
```python
# 1) gray 변환
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
tpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)  # template_img는 미리 준비
w, h = tpl_gray.shape[::-1]

# 2) 매칭(유사도 맵 생성)
res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)

# 3) 최댓값 위치가 “가장 비슷한 위치”
_, max_val, _, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
track_window = (top_left[0], top_left[1], w, h)  # (x,y,w,h)
```

- **왜 핵심?** 픽셀 기반 추적의 출발점이자, **한계(크기/회전/조명 변화 취약)**가 이후 기법 필요성을 만든다.
- **대표 개념:** “패치 유사도 기반 위치 탐색”.
- **다음 확장:** 멀티스케일(피라미드), 회전/스케일 불변 특징자 매칭(ORB/SIFT)로 연결.

---

## 2) 🔥 MeanShift: 색상 히스토그램(확률 분포) 기반 추적
### 2-1) 첫 프레임에서 ROI 히스토그램 등록
```python
# 초기 ROI (x, y, w, h) : 첫 프레임에서 객체 위치(하드코딩/ROI 선택)
x, y, w, h = 500, 300, 150, 100
roi = frame[y:y+h, x:x+w]

roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])  # H 채널
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
track_window = (x, y, w, h)
```

### 2-2) 매 프레임: BackProjection → MeanShift 업데이트
```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 확률맵(dst): “ROI 색이 나올 법한 곳”
dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

# meanShift: 확률이 가장 높은 쪽으로 창을 이동(크기는 고정)
_, track_window = cv2.meanShift(dst, track_window, term)
```

- **왜 핵심?** 객체를 “픽셀”이 아니라 **색 분포(확률)**로 보고 추적하는 관점 전환의 대표 코드.
- **대표 개념:** Histogram + BackProjection + 반복 수렴(Iteration).
- **다음 확장:** 스케일 변화/회전이 생기면 **CamShift**로 자연스럽게 확장.

---

## 3) 🔥 CamShift: MeanShift + 창 크기/회전까지 자동 적응
```python
# (선행) roi_hist, term, track_window는 MeanShift와 동일하게 준비되어 있다고 가정
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# (선택) 마스크로 노이즈(채도/명도 낮은 픽셀) 제거하면 안정적
mask = cv2.inRange(hsv, np.array((0., 70., 70.)), np.array((180., 200., 200.)))

dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
dst = dst * mask

rotated_rect, track_window = cv2.CamShift(dst, track_window, term)
# rotated_rect: (center, (w,h), angle)  ← 회전/스케일까지 포함
```

- **왜 핵심?** MeanShift의 약점(탐색창 크기 고정)을 보완하여 **거리 변화/회전**에 대응.
- **대표 개념:** “추적 창도 객체처럼 적응(Adaptive)한다”.
- **다음 확장:** CamShift ROI 내부에서 LK Optical Flow로 특징점 추적 결합(강건성 향상).

---

## 4) 🔥 Lucas–Kanade Sparse Optical Flow: “특징점이 어디로 이동했나”
### 4-1) 첫 프레임 특징점(코너) 검출
```python
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
```

### 4-2) 다음 프레임에서 특징점 이동 추정
```python
ret, frame2 = cap.read()
frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

p1, st, err = cv2.calcOpticalFlowPyrLK(
    old_gray, frame_gray, p0, None, **lk_params
)

good_new = p1[st == 1]
good_old = p0[st == 1]

# 업데이트(다음 루프 준비)
old_gray = frame_gray.copy()
p0 = good_new.reshape(-1, 1, 2)
```

- **왜 핵심?** “유사도 탐색”이 아니라 **움직임(변위)을 직접 계산**하는 추적의 중심.
- **대표 개념:** 밝기 보존/작은 이동/국소성 가정 + 피라미드(coarse-to-fine).
- **다음 확장:** Outlier 제거(RANSAC), 특징점 재검출, 객체 박스/마스크 결합으로 객체 단위 추적.

---

## 5) 🔥 Farneback Dense Optical Flow: “모든 픽셀의 흐름(Flow field)”
```python
prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ret, frame2 = cap.read()
gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(
    prev, gray, None,
    0.5, 3, 15, 3, 5, 1.1,
    cv2.OPTFLOW_FARNEBACK_GAUSSIAN
)  # flow.shape = (H, W, 2) : (dx, dy)

prev = gray
```

- **왜 핵심?** Sparse(LK)와 대비되는 “장면 전체 움직임” 추정의 대표 기법.
- **대표 개념:** Dense flow = 픽셀 단위 벡터장(u,v).
- **다음 확장:** magnitude/angle 분석, 배경/전경 분리, 행동 인식, 안정화, 딥러닝 RAFT로 연결.

---
