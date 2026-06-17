# PidTrace
본 프로젝트는 P&ID 도면 이미지에서 특정 배관의 시작점(FROM)과 끝점(TO)을 찾기 위한 프로토타입 프로젝트입니다. VLM(Vision-Language Model)과 이미지 처리 파이프라인을 결합하여 자연어 질문 기반으로 배관 경로를 추적합니다.

## 설정

1. `.env.example`를 `.env`로 복사합니다.

```bash
copy .env.example .env
```

2. `.env`에서 기본 경로를 확인합니다.

```env
INPUT_IMAGE_PATH=data/input/sample_pid.png ####
OUTPUT_DIR=data/output
```

3. 분석할 이미지를 `data/input/` 폴더에 넣습니다.

예:

- `data/input/sample_pid.png`
- `data/input/image.png`

중요:

- `INPUT_IMAGE_PATH`는 **이미지 파일명을 따로 넘기지 않았을 때 사용하는 기본값**입니다.
- 실행할 때 이미지 파일명을 같이 넘기면, 그 파일이 우선 사용됩니다.

## 설치

### 1. conda 환경 활성화

```bash
conda activate `프로젝트명`
```

환경이 아직 없다면:

```bash
conda env create -f environment.yml
conda activate `프로젝트명`
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 로컬 VLM 사용 시 Ollama 준비

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

기본 설정은 로컬 `Ollama`를 사용합니다.

## 실행 방법

### 권장 방식: 이미지 파일명 + 질문문장 함께 입력

```bash
python -m app.main sample_pid.png "이 이미지의 배관 200-P-310226-NB01-PP 의 시작과 끝 설비를 알려줘"
```

```bash
python -m app.main 3sample_pid.png "이 이미지의 배관 FV-3-3040 의 시작과 끝 설비를 알려줘"
```

이 경우 프로그램은:

- 첫 번째 인자로 받은 이미지 파일명을 사용하고
- 두 번째 인자인 질문 문장에서 배관명을 추출해 분석합니다.

상대 경로가 아닌 파일명만 넣어도 자동으로 `data/input/` 아래에서 찾습니다.

### 대체 방식: `.env`의 기본 이미지 사용

```bash
python -m app.main "이 이미지의 배관 200-P-310226-NB01-PP 의 시작과 끝 설비를 알려줘"
```

이 경우에는 `.env`의 `INPUT_IMAGE_PATH`에 설정된 이미지를 사용합니다.

## 실행 결과

~~- 콘솔 로그 출력~~
~~- 최종 답변 출력~~
- `data/output/result.json`
- `data/output/overlays/overlay.png`

## 1

- P&ID 이미지에서 특정 배관의 FROM/TO를 찾는 전체 파이프라인 프로토타입을 구현했습니다.
- 워크플로를 `OCR -> 설비 후보 추출 -> 배관 선분 추출 -> 타깃 배관 선택 -> 경로 추적 -> 오버레이 생성 -> VLM 추론` 순서로 연결했습니다.
- 사용자가 자연어 질문을 입력하면, 코드가 질문에서 배관명을 추출해 내부적으로 타깃 배관으로 사용하도록 구성했습니다.
- 최종 답변이 한국어 설명형 문장으로 출력되도록 프롬프트를 설계했습니다.
- `FROM`이 포함된 라벨은 시작점, `TO`가 포함된 라벨은 끝점으로 해석하도록 규칙을 반영했습니다.
- 타깃 배관을 찾지 못했을 때 VLM이 임의로 답하지 않도록 안전장치를 추가했습니다.
- `README`의 템플릿 생성
- `E-3118`, `EA-3114`, `E-3111` 같은 설비 태그도 일부는 인식되지만, 아직 일관되게 잡히지는 않습니다.
- `sample_pid.png` 같은 실제 도면에서는 `200-P-310225-NB01-HC`처럼 긴 배관명을 안정적으로 인식하기가 어렵습니다.

## 2

- 질문 문장 기반으로 실행할 수 있도록 입력 방식을 정리했습니다.
- 이미지 파일명을 실행 인자로 직접 넘길 수 있도록 구성했습니다.
- `FROM`, `TO` 라벨을 endpoint 힌트로 해석하도록 개선했습니다.
- OCR 정확도를 높이기 위해 업스케일, 샤프닝, threshold, 회전 OCR, 얇은 배관 라벨용 crop-upscale OCR을 추가했습니다.
     - 출력 텍스트 갯수 약 17.7퍼센트 증가(424 -> 499)
        - `sample_pid.png`에 대해서는 여전히 결과 도출x.
- 현재 코드는 `TO` 라벨 자체를 endpoint로 설정하지 못하고, 해당 위치 주변의 설비를 endpoint로 설정합니다.
    - fuzzy match를 도입해 해결(`3sample_pid.png`의 결과)

## 아직 어려운 부분

- 현재 프로젝트는 고정밀 실서비스보다는 “동작하는 프로토타입”에 가깝습니다.

## PaddleOCR 시도 및 롤백

- 작은 글씨 인식을 개선하기 위해 OCR 백엔드를 `PaddleOCR`로 교체하는 작업도 진행했습니다.
- 하지만 PaddleOCR의 런타임 호환성 문제와 버전별 API 차이로 인해 실제 추론 과정에서 오류가 발생했습니다.
- 그 결과 초기화 실패, 추론 중 예외, OCR 결과가 비는 문제 등이 발생했습니다.
- 저장소를 안정적으로 실행 가능한 상태로 유지하기 위해 최종 버전은 다시 `EasyOCR` 기반으로 롤백했습니다.
