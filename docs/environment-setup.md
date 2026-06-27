# 환경 구축 기록: AMD GPU에서 PyTorch 학습 (WSL2 + ROCm + ROCDXG)

이 문서는 **AMD Radeon RX 7800 XT**에서 PyTorch GPU 학습 환경을 구축한 전 과정과,
그 과정에서 마주친 문제·원인·해결을 기록합니다.

> **한 줄 요약**: AMD GPU + Windows에서는 CUDA를 쓸 수 없어 WSL2 + ROCm로 구성한다.
> 일반 ROCm 설치만으로는 WSL에서 GPU가 잡히지 않으며, **`librocdxg`(ROCDXG)** 라는
> WSL↔Windows 드라이버 연결 계층을 별도로 설치하고 **`HSA_ENABLE_DXG_DETECTION=1`** 을
> 설정해야 GPU가 인식된다.

## 1. 출발점과 제약

| 항목 | 내용 |
|---|---|
| GPU | AMD Radeon RX 7800 XT (RDNA3, **gfx1101**, VRAM 16GB) |
| OS | Windows 11 |
| Windows 드라이버 | AMD Adrenalin **26.2.2** (드라이버 파일 버전 32.0.23027.2005, 2026-02) |

**가장 큰 제약: AMD GPU는 NVIDIA CUDA를 쓸 수 없다.**
대부분의 PyTorch 자료가 CUDA 전제라, AMD에서는 다음 중 하나를 택해야 한다.

- **WSL2 + ROCm** — RDNA3 지원, 거의 네이티브 성능. 셋업이 번거로움. → **채택**
- torch-directml — 설치는 쉬우나 느리고 구버전 torch에 고정
- CPU — 항상 동작하나 느림

학습 성능이 중요하므로 **WSL2 + ROCm**를 선택했다.

## 2. 사전 점검 결과

| 항목 | 상태 |
|---|---|
| WSL2 | Ubuntu 24.04.4 LTS (이미 설치됨), RAM 15GB 할당 |
| GPU passthrough | `/dev/dxg` 존재 (DirectX 경로 OK) |
| ROCm 런타임 | 미설치 (`/opt/rocm` 없음) |
| 디스크 | 950GB 여유 |
| Windows 드라이버 | Adrenalin 26.2.2 (ROCm 7.2.1이 요구하는 버전과 일치) |

**버전 매칭이 핵심.** ROCm 7.2.1 ↔ Adrenalin 26.2.2 ↔ librocdxg 1.2.0 조합이 서로 맞아야 한다.
다행히 설치된 드라이버가 26.2.2라 최신 ROCm 7.2.1을 목표로 잡았다.

## 3. 설치 절차 (최종)

> sudo가 필요한 단계는 WSL 터미널에서 직접 실행한다(비대화형 환경에서는 sudo 비밀번호 입력 불가).

```bash
# (1) ROCm 7.2.1 설치
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.1.70201-1_all.deb
sudo amdgpu-install -y --usecase=rocm --no-dkms     # WSL이므로 --no-dkms (커널 모듈 제외)

# (2) librocdxg 설치 — WSL에서 GPU를 잡기 위한 핵심
#     github.com/ROCm/librocdxg releases 의 사전 빌드 deb
sudo apt install -y ./rocdxg-roct_1.2.0_amd64.deb

# (3) PyTorch (ROCm 7.2.1 WSL 휠) — conda env 안에서
conda create -y -n tf python=3.12
conda activate tf
pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

**GPU 코드 실행 시 반드시 환경변수 설정** (ROCm 7.13 미만 WSL 요구):

```bash
HSA_ENABLE_DXG_DETECTION=1 python my_script.py
```

저장소의 스크립트로 (1)·(2)·(3 검증)을 재현할 수 있다:
[`scripts/00_install_rocm.sh`](../scripts/00_install_rocm.sh),
[`01_install_rocdxg.sh`](../scripts/01_install_rocdxg.sh),
[`02_verify_gpu.sh`](../scripts/02_verify_gpu.sh).

## 4. 트러블슈팅 사례

실제로 막혔던 지점들을 순서대로 기록한다. (포트폴리오용 핵심)

### 4.1 Python 3.14는 PyTorch에 너무 최신

- **증상**: Windows에 Python 3.14만 설치되어 있었음.
- **원인**: PyTorch 휠은 보통 Python 3.11~3.13까지만 제공(3.14 미지원).
- **해결**: WSL Ubuntu의 시스템 Python 3.12를 사용. 단 venv는 아래 4.2 문제로 conda로 우회.

### 4.2 시스템 Python venv가 막힘 → conda 사용

- **증상**: `python3.12 -m venv` 가 pip 부트스트랩에서 실패.
- **원인**: 우분투 시스템 Python 3.12에 `ensurepip` 모듈이 없음(별도 `python3.12-venv` apt 패키지 필요, sudo 요구).
- **해결**: WSL에 이미 있던 miniconda로 `conda create -n tf python=3.12`. sudo 없이 격리 환경 확보.

### 4.3 `amdgpu-install`에 `wsl` usecase가 없음

- **증상**: `amdgpu-install --usecase=wsl,rocm` 실행 시
  `ERROR: Usecase implementation 'wsl' is not supported or invalid`.
- **원인**: 구버전 ROCm(6.x) 문서의 `wsl,rocm` 방식은 더 이상 유효하지 않음.
  ROCm 7.2.x부터 WSL 지원이 ROCDXG로 재편되며 `wsl` usecase가 제거됨
  (`amdgpu-install --list-usecase`로 확인 — `rocm`만 존재).
- **해결**: `--usecase=rocm --no-dkms` 사용.

### 4.4 PyTorch 휠이 AMD 전용 triton을 요구

- **증상**: AMD torch 휠 설치 중
  `ERROR: No matching distribution found for triton==3.5.1+rocm7.2.1...` (PyPI에 없음).
- **원인**: AMD WSL용 torch 휠이 같은 repo.radeon.com에 있는 **전용 triton 휠**을 하드 의존성으로 가짐.
- **해결**: torch와 같은 디렉터리(`rocm-rel-7.2.1`)의 triton 휠을 함께 설치.

### 4.5 (핵심) GPU 미탐지 — `hipErrorNoDevice`

- **증상**: 설치는 됐는데 `torch.cuda.is_available()` 가 **False**.
  상세 로그(`AMD_LOG_LEVEL=3`)에서 `hipGetDeviceCount ... Returned hipErrorNoDevice`.
  `rocminfo`도 GPU 에이전트를 0개로 보고. `HSA_OVERRIDE_GFX_VERSION`도 효과 없음.
- **진단 과정**:
  - `/dev/dxg`는 존재하지만 `/dev/kfd`·`/dev/dri`는 없음 → WSL은 KFD가 아닌 dxg 경로를 씀.
  - 설치된 `libhsa-runtime64.so`는 **네이티브 Linux(KFD 기반)** 런타임 → `/dev/kfd`가 없는 WSL에선 장치를 못 찾음.
  - `/opt/rocm/lib`와 `/usr/lib/wsl/lib` 어디에도 WSL용 컴퓨트 라이브러리(`librocdxg`)가 없음.
- **원인**: `amdgpu-install --usecase=rocm`은 **네이티브 ROCm만** 설치한다.
  WSL에서 GPU 컴퓨트를 가능하게 하는 **`librocdxg`(ROCDXG)** 는 별도 패키지로,
  Linux ROCm 런타임과 Windows GPU 드라이버 사이의 **변환 계층** 역할을 한다.
- **해결**: GitHub `ROCm/librocdxg`의 사전 빌드 deb(`rocdxg-roct_1.2.0_amd64.deb`) 설치 →
  `/opt/rocm/lib/librocdxg.so` 생성. RX 7800 XT는 ROCDXG 1.2.0 + ROCm 7.2.x 공식 지원 목록에 포함.

### 4.6 ROCDXG에 필요한 환경변수

- **증상**: librocdxg 설치 후에도 환경변수가 없으면 탐지가 불안정.
- **원인**: ROCm 7.13 미만에서는 ROCDXG 탐지를 위해 `HSA_ENABLE_DXG_DETECTION=1` 가 **필수**(7.13부터 불필요).
- **해결**: GPU 실행 시 `HSA_ENABLE_DXG_DETECTION=1` 설정(검증 스크립트가 자동 export).

## 5. 최종 검증 결과

`librocdxg` 설치 + `HSA_ENABLE_DXG_DETECTION=1` 적용 후:

```
rocminfo  → Marketing Name: AMD Radeon RX 7800 XT, Name: gfx1101  (에이전트 인식)
torch.cuda.is_available()  → True
device  → AMD Radeon RX 7800 XT
```

학습에 필요한 연산이 모두 GPU에서 정상 동작함을 확인:

| 검증 항목 | 결과 |
|---|---|
| 학습 루프 (forward + backward + AdamW) | loss 1.05 → 0.006 (정상 감소) |
| bf16 autocast (혼합정밀) | OK |
| VRAM | 15.8 GiB 인식 |
| matmul 4096³ (fp32) | ~9.1 TFLOP/s |

> gfx1101은 ROCm 7.2.1에서 네이티브 지원되어 `HSA_OVERRIDE_GFX_VERSION` 우회가 필요 없었다.
> (미탐지 시에만 `HSA_OVERRIDE_GFX_VERSION=11.0.0` 시도가 일반적 대처법)

## 6. 핵심 교훈

1. **AMD + WSL의 함정은 ROCm 설치가 아니라 `librocdxg`다.** 일반 ROCm 설치 가이드만 따르면
   `hipErrorNoDevice`에서 막히기 쉽다. WSL은 `/dev/dxg` 경로를 쓰며 ROCDXG가 그 다리를 놓는다.
2. **버전 삼각 매칭**: ROCm ↔ Adrenalin 드라이버 ↔ librocdxg 버전이 서로 맞아야 한다.
3. **공식 문서가 JS 렌더링이라 추출이 어려운 경우**, 패키지 저장소(`repo.radeon.com`)와
   GitHub 릴리스를 직접 조회해 정확한 파일명·버전을 확인하는 편이 빠르다.

## 참고 링크

- [ROCm/librocdxg (GitHub)](https://github.com/ROCm/librocdxg)
- [Install Radeon software for WSL with ROCm (AMD 공식)](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html)
- [Install PyTorch for ROCm (WSL)](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html)
- [AMD ROCDXG production support — Phoronix](https://www.phoronix.com/news/AMD-WSL-ROCm-ROCDXG)
