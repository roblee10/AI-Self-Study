# Pytorch nn.Module 컨볼루션

```python
class ConvNet(nn.Module):
    def __init__(self, args, class_n=7):
        super().__init() # 클래스 상속, nn.Module 클래스의 속성과 메소드를 상속받는다

    self.model = nn.Sequential(
        # [32,3,224,224] -> [32,32,55,55] (개수,채널,가로,세로)
        nn.Conv2d(in_channels= 3, out_channels=32, kernel_size=8, stride=4),
        nn.BatchNorm2d(32), #(32 는 채널), 값들 정규화
        nn.ReLU(), # 음수는0, 양수는 입력값 그대로 사용하는 Activation Function
        nn.MaxPool2d((2,2)) # Sub-sampling 기술, 2X2 필터로 최대값만 샘플링, Stride = None 이 default 이므로 그냥 저장
        # [32,32,55,55] -> [32,32,27,27]
        
        # [32,32,27,27] -> [32,64,11,11]
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size =7, stride = 2)
        nn.BatchNorm2d(64)
        nn.ReLU()
        nn.MaxPool2d((2,2)) # [32,64,5,5]
    )
    self.fc == nn.Linear(1600, class_n)

    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

# 컨볼루션 필터 계산

$$
O = \frac{I-K+2P}{S} +1
$$

O = 출력 이미지 크기

I = 입력 이미지 크기

K = 컨볼루션 레이어 커널의 사이즈

N = 출력 채널 개수

S = 컨볼루션 연산의 Stride

P = Padding 사이즈

EX1) 

- input = torch.Tensor(24,3,224,224) → 24(개)X3(채널)X224X224(이미지 크기)
- conv = torch.nn.Conv2d(in_channels= 3, out_channels=32, kernel_size=8, stride=4)

I = 224 (입력 이미지 크기)

K =  8 (컨볼루션 커널 사이즈)

N = 32 (출력 채널 개수)

S = 4 (Stride)

P = 0 (default padding)

$$
O = \frac{224-8+2*0}{4} +1 = 55
$$

- output = torch.Tensor(24,32,55,55)
- 96 = 8X8(커널크기) X 32 (채널)

# Batch Normalization 2d

- 문제점: Internal Covariate shift → Batch 별로 분포가 달라지게 됨
- 방법 : 정규 분포(평균 0 분사1) 로 Batch 분포 정규화하
- 효과 :
    - Internal Covariate shift 개선
    - Activation에 들어가게 되는 input range 를 제한
    - learning rate 의 자유도 개선
    - overfitting에 강건해짐
- 한계 :
    - Batch 의 크기가 너무 작으면 데이터 분포 표현을 제대로 못함
    - Batch 크기가 너무 크면 multi modal 형태의  gaussian mixture 모델 형태가 나타날 수 있고, 병렬 연산에 비효율적

사용법

- torch.nn.BatchNorm2d(N, C, H, W)
- N = Batch의 크기
- C = Channel
- H = Height
- W = Width
- C 채널은 필수 입력, 채널값을 기준으로 연산되기 때문

# RELU

문제점: 

- 기존 활성화 함수( linear 한 입력값을 non-linear 한 출력값으로 만들어주는 함수, XOR 같은 non-linear 문제 해결하게 해줌) 인 sigmoid의 출력값이 너무 작아 제대로 학습이 안되는 gradient vanishing 문제 발생

방법: 

- RELU 는 + 신호는 그대로, - 신호는 0으로 치환

효과: 

- 음수는 0, 양수는 입력값 그대로 사용
- 미분값은 0 또는 1
- 출력값의 범위가 넓어 gradient vanishing 문제 발생하지 않음
- 공식이 단순해 속도가 빠름

한계: 

- 음수에 대해 모두 0 처리 → 음수는 학습이 안됨 → 죽은 뉴런들이 발생하게됨

사용법:

- torch.nn.ReLU()

## MAX Pooling Layer

문제점:

- conv layer 는 feature map 계산
- activation function 은 non-linearity 부여
- 연산 시 RAM 메모리를 많이 잡아먹음
- pooling 은 sub-sampling을 하여 압축하는 과정

방법 (Max pooling 기준)

- activation function을 거친 feature map 값을 filter 크기와 stride에 따라 최대값만 샘플링
- EX) 4X4 feature map 에 2X2 max pooling, stride 2 적용
    - 4x4 feature map 의 겹치지 않는 각 2X2 영역 중 최대값만 샘플링하게됨
    - 따라서 2X2 feature map 이 결과로 나옴

효과

- input 텐서 압축
- overfitting 방지 → 쓸데없는 parameter 제거
- 특징을 더 잘 추출하게 됨

한계

- 이미지의 정보를 많이 누락시키게 됨

사용법

- torch.nn.MaxPool2d((2,2))
- (2,2) 의 필터 사이즈 maxpooling
- default stride = None 이므로 영역 오버랩 없이 2X2 크기 maxpooling 하게된다.

## Linear Function

문제점

- 컨볼루션으로 얻은 텐서 feature는 너무 고차원
- 분류를 위해서 이를 1차원으로 만들어 줄 필요가 있음

방법

- input channel * feature row * feature col 크기의 입력을 출력 사이즈로 압축하여 나타낸다
- EX) [32,64,5,5] 의 32개 이미지 feature 를 32개의 10 사이즈의 1차원 리스트로 변환
- [32, 64X5X5]= [32,1600] 으로 변환 → fully connected layer 통과 → [32,10]

효과

- feature map을 이용해 특정 라벨에 대한 확률을 계산
- 공간 정보의 압축

한계점

- 공간 정보의 손실
- overfitting 발생
- 학습하지 않은 레이블에 대해선 분류 성능 처참

사용법

- torch.nn.Linear(input dimension, output dimension)
- torch.nn.Linear(1600,10): [32,64,5,5] to [32,1600] → [32,10]

## Forward

- 모델이 학습데이터를 입력받아서 forward propagation 진행시키는 함수
- 모델을 설계한 순서대로 데이터가 통과된다.

```python
def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

- `x.view(x.size(0), -1)` 함수는 입력 받은 텐서의 크기를 변경하는 함수이다. 여기서 x.size(0) 은 배치 사이즈를 나타내고, -1 은 나머지 값을 자동으로 채우기 위함이다. 따라서 입력 받은 텐서의 크기를 배치 사이즈로 고정하고 나머지 차원의 크기를 자동으로 채워넣어준다.

- self.fc(x)은 모델의 마지막 층으로, 입력 받은 데이터 x를 Fully Connected Layer(완전 연결 계층)을 통과시켜 출력값을 반환하는 함수입니다.