# AI응용_5차시_핵심코드.md
> 주제: CNN 기반 분류 + 전이학습(Transfer Learning) + 대표 아키텍처(LeNet/AlexNet/VGG/GoogLeNet)

---

## 0) 🔥 공통: 데이터 파이프라인 (Custom Dataset → DataLoader)
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder("data/train", transform=train_tf)
val_ds   = datasets.ImageFolder("data/val",   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

num_classes = len(train_ds.classes)
```

- **왜 핵심?** 5차시 목표(커스텀 데이터셋/전이학습)는 결국 “데이터를 표준 형태로 공급”하는 파이프라인에서 시작.
- **대표 개념:** Resize/Normalize로 입력 분포 정렬 → 학습 안정화.
- **다음 확장:** Augmentation 강도 조절, 불균형 처리(WeightedSampler/가중치 손실).

---

## 1) 🔥 LeNet: CNN 구조 이해의 기준 모델
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      # (N,1,32,32) -> (N,6,28,28)
        self.pool  = nn.AvgPool2d(2, 2)                  # -> (N,6,14,14)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)     # -> (N,16,10,10)
        # pool -> (N,16,5,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)          # (N, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                  # logits
        return x
```

- **왜 핵심?** Conv→Pool→FC로 “특징 추출→요약→분류” 흐름을 가장 단순하게 고정하는 기준점.
- **대표 개념:** 합성곱(특징) / 풀링(요약·불변성) / FC(결정).
- **다음 확장:** 더 깊은 모델(VGG/ResNet) + BN/Dropout 추가.

---

## 2) 🔥 학습 루프: “CNN 분류가 실제로 학습되는 방식”
```python
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def run_one_epoch(model, loader, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total
```

- **왜 핵심?** 모델이 있어도 “loss/optimizer/train-eval 모드”가 없으면 학습이 성립하지 않음.
- **대표 개념:** logits → CrossEntropyLoss(내부 softmax 개념)로 확률적 분류 학습.
- **다음 확장:** scheduler, early stopping, best model 저장/복원.

---

## 3) 🔥 Transfer Learning(AlexNet): “사전학습 모델을 내 데이터에 맞춘다”
### 3-1) 로드 + (선택) Freeze + 분류기 교체
```python
import torchvision.models as models

model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# (선택) 특징추출기로만 쓰고 싶으면 freeze
for p in model.features.parameters():
    p.requires_grad = False

# 분류기 교체 (마지막 Linear만 내 클래스 수로)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)
```

### 3-2) 학습할 파라미터만 옵티마이저에 등록
```python
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
```

- **왜 핵심?** 5차시의 실무 축: 적은 데이터로 빠르게 성능을 내는 가장 표준적인 방법.
- **대표 개념:** backbone(재사용 특징) + head(교체 학습) 분리.
- **다음 확장:** partial fine-tune(일부 블록 unfreeze), 파라미터 그룹별 학습률(헤드↑ 백본↓).

---

## 4) 🔥 Pretrained VGG13 Inference: “학습이 아니라 사용(추론)”
```python
from PIL import Image
import torch
import torchvision.models as models

weights = models.VGG13_Weights.DEFAULT
model = models.vgg13(weights=weights).to(device)
model.eval()

preprocess = weights.transforms()

img = Image.open("test.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0).to(device)  # (1,3,224,224)

with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1).item()
```

- **왜 핵심?** “사전학습 모델을 불러와 바로 분류한다”는 분류 앱/데모의 핵심 흐름.
- **대표 개념:** eval/no_grad + (weights.transforms로) 전처리 일관성 유지.
- **다음 확장:** top-k, confidence(softmax), 배치 추론, TorchScript/ONNX.

---

## 5) 🔥 GoogLeNet/Inception: “성능 vs 효율을 구조로 해결”
```python
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_ch, c1, c3r, c3, c5r, c5, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, c3r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3r, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, c5r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5r, c5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)
```

- **왜 핵심?** Inception의 요지는 “병렬 필터 + 1×1 차원축소”로 파라미터/연산량을 줄이면서 성능 유지.
- **대표 개념:** 병렬 분기(다중 스케일 특징) + 1×1 conv(차원 축소).
- **다음 확장:** ResNet(skip), MobileNet(depthwise separable)로 자연스럽게 이어짐.

---
